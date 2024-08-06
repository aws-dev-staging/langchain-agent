import re
import os
import json
import time
import boto3
import pdfrw
import string
import random
import difflib
import datetime

#from chat import Chat
from boto3.dynamodb.conditions import Key
#from langchain.llms.bedrock import Bedrock
#from langchain.chains import ConversationChain
#from home_warranty_agent import HomeWarrantyAgent

# Create reference to DynamoDB tables and S3 bucket
users_table_name = os.environ['USERS_TABLE_NAME']
claims_table_name = os.environ['CLAIMS_TABLE_NAME']
home_warranty_quote_requests_table_name = os.environ['INSURACE_QUOTE_REQUESTS_TABLE_NAME']
s3_artifact_bucket = os.environ['S3_ARTIFACT_BUCKET_NAME']

# Instantiate boto3 clients and resources
boto3_session = boto3.Session(region_name=os.environ['AWS_REGION'])
dynamodb = boto3.resource('dynamodb',region_name=os.environ['AWS_REGION'])
s3_client = boto3.client('s3',region_name=os.environ['AWS_REGION'],config=boto3.session.Config(signature_version='s3v4',))
s3_object = boto3.resource('s3')
bedrock_client = boto3_session.client(service_name="bedrock-runtime")
agent_runtime_client = boto3_session.client('bedrock-agent-runtime')

# --- Lex v2 request/response helpers (https://docs.aws.amazon.com/lexv2/latest/dg/lambda-response-format.html) ---

def elicit_slot(session_attributes, active_contexts, intent, slot_to_elicit, message):
    """
    Constructs a response to elicit a specific Amazon Lex intent slot value from the user during conversation.
    """
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'ElicitSlot',
                'slotToElicit': slot_to_elicit 
            },
            'intent': intent,
        },
        'messages': [{
            "contentType": "PlainText",
            "content": message,
        }]
    }

    return response

def elicit_intent(intent_request, session_attributes, message):
    """
    Constructs a response to elicit the user's intent during conversation.
    """
    response = {
        'sessionState': {
            'dialogAction': {
                'type': 'ElicitIntent'
            },
            'sessionAttributes': session_attributes
        },
        'messages': [
            {
                'contentType': 'PlainText', 
                'content': message
            },
            {
                'contentType': 'ImageResponseCard',
                'imageResponseCard': {
                    "buttons": [
                        {
                            "text": "Home Warranty Quote",
                            "value": "Home Warranty Quote"
                        },
                        {
                            "text": "Summarize Claims",
                            "value": "Summarize Claims"
                        },
                        {
                            "text": "Ask GenAI",
                            "value": "What kind of questions can the assistant answer?"
                        }
                    ],
                    "title": "How can I help you?"
                }
            }     
        ]
    }

    return response

def delegate(session_attributes, active_contexts, intent, message):
    """
    Delegates the conversation back to the system for handling.
    """
    response = {
        'sessionState': {
            'activeContexts':[{
                'name': 'intentContext',
                'contextAttributes': active_contexts,
                'timeToLive': {
                    'timeToLiveInSeconds': 86400,
                    'turnsToLive': 20
                }
            }],
            'sessionAttributes': session_attributes,
            'dialogAction': {
                'type': 'Delegate',
            },
            'intent': intent,
        },
        'messages': [{'contentType': 'PlainText', 'content': message}]
    }

    return response

def build_slot(intent_request, slot_to_build, slot_value):
    """
    Builds a slot with a specified slot value for the given intent_request.
    """
    intent_request['sessionState']['intent']['slots'][slot_to_build] = {
        'shape': 'Scalar', 'value': 
        {
            'originalValue': slot_value, 'resolvedValues': [slot_value], 
            'interpretedValue': slot_value
        }
    }

def build_validation_result(isvalid, violated_slot, message_content):
    """
    Constructs a validation result indicating whether a slot value is valid, along with any violated slot and an accompanying message.
    """
    return {
        'isValid': isvalid,
        'violatedSlot': violated_slot,
        'message': message_content
    }
    
# --- Utility helper functions ---

def isvalid_slot_value(value, slot_value_list):
    """
    Compares 'value' to 'slot_value_list' based on 'similarity_threshold'.
    """
    similarity_threshold = 0.5

    # Split the value into components
    value_components = [v.strip() for v in value.split(',')]

    similarity_scores = []

    # Calculate similarity for each component of value
    for component in value_components:
        scores = [difflib.SequenceMatcher(None, component.lower(), ref_value.lower()).ratio() for ref_value in slot_value_list]
        similarity_scores.extend(scores)

    print(f"value: {value}\nslot_value_list: {slot_value_list}\nsimilarity_scores: {similarity_scores}")
    
    # Check if any similarity score meets or exceeds the threshold
    return any(score >= similarity_threshold for score in similarity_scores)

def parse_date(date_str):
    """
    Parses 'date_str' into a datetime object before performing date arithmetic.
    """
    try:
        return datetime.datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

def create_presigned_url(bucket_name, object_name, expiration=600):
    """
    Generate a presigned URL for the S3 object.
    """
    try:
        response = s3_client.generate_presigned_url('get_object', Params={'Bucket': bucket_name, 'Key': object_name}, ExpiresIn=expiration)
        return response
    except Exception as e:
        print(f"Error creating S3 presigned URL: {e}")

    return None

def try_ex(value):
    """
    Safely access slots dictionary values.
    """
    if value and value.get('value'):
        return value['value'].get('interpretedValue') or value['value'].get('originalValue')
    return None

def get_user_by_policy_id(policy_id):
    """
    Retrieves user information based on the provided policyId using a GSI.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Set up the query parameters for the GSI
        params = {
            'IndexName': 'PolicyIdIndex',
            'KeyConditionExpression': 'PolicyId = :pid',
            'ExpressionAttributeValues': {
                ':pid': policy_id
            }
        }

        # Execute the query and get the result
        response = users_table.query(**params)

        # Check if any items were returned
        if response['Count'] > 0:
            return response['Items']
        else:
            print("No user found with the given policyId")

    except Exception as e:
        print(f"Error retrieving user by policyId: {e}")
    
    return None 

# --- Intent fulfillment functions ---

def isvalid_pin(username, pin):
    """
    Validates the user-provided PIN using a DynamoDB table lookup.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Query the table using the partition key
        response = users_table.query(
            KeyConditionExpression=Key('UserName').eq(username)
        )

        # Iterate over the items returned in the response
        if len(response['Items']) > 0:
            pin_to_compare = int(response['Items'][0]['Pin'])
            # Check if the password in the item matches the specified password
            if pin_to_compare == int(pin):
                return True

        print("PIN did not match")
        return False

    except Exception as e:
        print(f"Error validating PIN: {e}")
        return e

def isvalid_username(username):
    """
    Validates the user-provided username exists in the 'claims_table_name' DynamoDB table.
    """
    users_table = dynamodb.Table(users_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': 'UserName = :c',
            'ExpressionAttributeValues': {
                ':c': username
            }
        }

        # Execute the query and get the result
        response = users_table.query(**params)     

        # Check if any items were returned
        if response['Count'] != 0:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error validating username: {e}")
        return e

def validate_pin(intent_request, username, pin):
    """
    Elicits and validates user input values for username and PIN. Invoked as part of 'verify_identity' intent fulfillment.
    """
    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
        session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
        session_attributes['UserName'] = username
        intent_request['sessionState']['sessionAttributes']['UserName'] = username
    else:
        return build_validation_result(
            False,
            'UserName',
            'Our records indicate there are no accounts belonging to that username. Please try again.'
        )

    if pin is not None:
        if  not isvalid_pin(username, pin):
            return build_validation_result(
                False,
                'Pin',
                'You have entered an incorrect PIN. Please try again.'.format(pin)
            )
    else:
        message = "Thank you for choosing First American Home Warranty, {}. Please confirm your 4-digit PIN before we proceed.".format(username)
        return build_validation_result(
            False,
            'Pin',
            message
        )

    return {'isValid': True}

def verify_identity(intent_request):
    """
    Performs dialog management and fulfillment for username verification.
    Beyond fulfillment, the implementation for this intent demonstrates the following:
    1) Use of elicitSlot in slot validation and re-prompting.
    2) Use of sessionAttributes {UserName} to pass information that can be used to guide conversation.
    """
    slots = intent_request['sessionState']['intent']['slots']
    pin = try_ex(slots['Pin'])
    username=try_ex(slots['UserName'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    # Validate any slots which have been specified. If any are invalid, re-elicit for their value
    validation_result = validate_pin(intent_request, username, pin)
    session_attributes['UserName'] = username

    if not validation_result['isValid']:
        slots = intent_request['sessionState']['intent']['slots']
        slots[validation_result['violatedSlot']] = None

        return elicit_slot(
            session_attributes,
            active_contexts,
            intent_request['sessionState']['intent'],
            validation_result['violatedSlot'],
            validation_result['message']
        )
    else:
        if confirmation_status == 'None':
            # Query DDB for user information before offering intents
            users_table = dynamodb.Table(users_table_name)

            try:
                # Query the table using the partition key
                response = users_table.query(
                    KeyConditionExpression=Key('UserName').eq(username)
                )

                # Customize message based on home warranty details
                message_parts = []
                items = response['Items']

                for item in items:
                    policy_id = item.get('PolicyId')
                    property_type = item.get('PropertyType')
                    property_value = item.get('PropertyValue')
                    property_address = item.get('PropertyAddress', {})
                    plan_type = item.get('PlanType')
                    covered_items = item.get('CoveredItems', [])
                    deductible_amount = item.get('DeductibleAmount')
                    service_fee = item.get('ServiceFee')
                    coverage_description = item.get('CoverageDescription', '')
                    policy_start_date = item.get('PolicyStartDate')
                    policy_end_date = item.get('PolicyEndDate')

                    if policy_id:
                        session_attributes['PolicyId'] = policy_id
                        build_slot(intent_request, 'PolicyId', policy_id)
                        print(f"verify_identity session_attributes: {session_attributes}")
                        message_parts.append(f"Your home warranty policy ID is {policy_id}.")

                    if plan_type and property_type and property_value:
                        build_slot(intent_request, 'PlanType', plan_type)
                        build_slot(intent_request, 'PropertyType', property_type)
                        build_slot(intent_request, 'PropertyValue', property_value)
                        message_parts.append(f"\n\nYour {plan_type} covers a {property_type} valued at ${property_value:,}.")

                    if property_address:
                        address = ', '.join(filter(None, [
                            property_address.get('Street', ''),
                            property_address.get('City', ''),
                            property_address.get('State', ''),
                            property_address.get('Zip', '')
                        ]))

                        if address:
                            build_slot(intent_request, 'PropertyAddress', address)
                            message_parts.append(f"Located at {address}.")

                    if covered_items and coverage_description:
                        build_slot(intent_request, 'CoveredItems', covered_items)
                        build_slot(intent_request, 'CoverageDescription', coverage_description)
                        message_parts.append(f"{coverage_description}")

                    if deductible_amount and service_fee:
                        build_slot(intent_request, 'DeductibleAmount', deductible_amount)
                        build_slot(intent_request, 'ServiceFee', service_fee)
                        message_parts.append(f"\n\nYour deductible amount is ${deductible_amount:,} and your service fee is ${service_fee}.")

                    if policy_start_date and policy_end_date:
                        build_slot(intent_request, 'PolicyStartDate', policy_start_date)
                        build_slot(intent_request, 'PolicyEndDate', policy_end_date)
                        message_parts.append(f"\n\nThe policy started on {policy_start_date} and ends on {policy_end_date}.")

                message = ' '.join(message_parts)

                return elicit_intent(
                    intent_request,
                    session_attributes,
                    f'Thank you for confirming your username and PIN, {username}. {message}'
                )

            except Exception as e:
                print(f"Error querying DynamoDB: {e}")
                return e


def validate_home_warranty_quote(intent_request, slots):
    """
    Validates slot values specific to Home Warranty insurance.
    """
    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    session_id = intent_request['sessionId']

    username = try_ex(slots['UserName'])
    property_type = try_ex(slots['PropertyType'])
    plan_type = try_ex(slots['PlanType'])
    appliances = try_ex(slots['Appliances'])
    plumbing = try_ex(slots['Plumbing'])
    systems = try_ex(slots['Systems'])
    hvac = try_ex(slots['HVAC'])
    additional_limits = try_ex(slots['AdditionalLimits'])
    policy_start_date = try_ex(slots['PolicyStartDate'])
    policy_end_date = try_ex(slots['PolicyEndDate'])

    user_help_flag = False

    # Validate UserName
    if username is not None:
        if not isvalid_username(username):
            return build_validation_result(
                False,
                'UserName',
                'Our records indicate there is no profile belonging to the username, {}. Please enter a valid username'.format(username)
            )
    else:
        try:
            session_username = intent_request['sessionState']['sessionAttributes']['UserName']
            build_slot(intent_request, 'UserName', session_username)
        except KeyError:
            return build_validation_result(
                False,
                'UserName',
                'You have been logged out. Please start a new session.'
            )

    # Validate PropertyType
    property_type_list = ['Single-Family','Multi-Family', 'Condominium', 'Townhome', 'Mobile Home']
    property_types_str = ', '.join(property_type_list)
    if property_type is not None:
        if not isvalid_slot_value(property_type, property_type_list):
            prompt = f"The user was asked to specify the property type [{property_types_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
            bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
            if 'messages' in bedrock_response:
                message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                print(f"Extracted message content: {message_content}")
            else:
                message_content = "I couldn't find any relevant information."

            reply = f"{message_content}\n\nWhich type of property do you want to protect?\n\n[{property_types_str}]."
            return build_validation_result(False, 'PropertyType', reply)
    else:
        return build_validation_result(
            False,
            'PropertyType',
            f'Please specify the type of property you want to protect - \n\n[{property_types_str}].'
        )

    # Validate PlanType
    plan_type_list = ['Starter', 'Essential', 'Premium', 'Unsure']
    user_help_flag_list = ['Unsure']
    plan_type_list_str = "Starter, Essential, Premium"
    if plan_type is not None:
        if not isvalid_slot_value(plan_type, plan_type_list):
            prompt = f"The user was asked to select a plan type [{plan_type_list_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
            bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
            if 'messages' in bedrock_response:
                message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                print(f"Extracted message content: {message_content}")
            else:
                message_content = "I couldn't find any relevant information."

            reply = f"{message_content}\n\nWhich coverage plan would you like?\n\n[{plan_type_list_str}].\n\nRespond 'Unsure' if you would like assistance selecting your plan type."
            return build_validation_result(False, 'PlanType', reply)
        elif isvalid_slot_value(plan_type, user_help_flag_list):
            user_help_flag = True
    else:
        return build_validation_result(
            False,
            'PlanType',
            f"Please specify your desired coverage plan - \n\n[{plan_type_list_str}].\n\nRespond 'Unsure' if you would like assistance selecting your plan type."
        )

    # if PlanType is "unsure" elicit appliances, plumbing, systems, HVAC, and addition limits slot values
    print(f"user_help_flag: {user_help_flag}")
    if user_help_flag:

        # Validate Appliances
        appliances_list = ['Dishwasher', 'Refrigerator', 'Range Hood', 'Microwave', 'Oven', 'Washer', 'Dryer', 'Trash Compactor', 'All', 'None']
        appliances_list_str = ', '.join(appliances_list)
        if appliances is not None:
            if not isvalid_slot_value(appliances, appliances_list):
                prompt = f"The user was asked to specify the appliances ({appliances_list_str}) as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
                bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
                if 'messages' in bedrock_response:
                    message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                    print(f"Extracted message content: {message_content}")
                else:
                    message_content = "I couldn't find any relevant information."

                reply = f"{message_content}\n\nWhich appliances do you want covered?\n\n[{appliances_list_str}]."
                return build_validation_result(False, 'Appliances', reply)
        else:
            return build_validation_result(
                False,
                'Appliances',
                f'To help determine the best home warranty plan for you, please specify the appliances you want covered -\n\n[{appliances_list_str}].'
            )

        # Validate Plumbing
        plumbing_list = ['Plumbing', 'Plumbing Stoppages', 'Toilet Tanks, Bowls and Mechanisms', 'Water Heater', 'Garbage Disposal', 'Hose Bibbs', 'Install Ground Level Cleanout', 'Instant Hot Water Dispenser', 'Shower Head and Shower Arm', 'All', 'None']
        plumbing_list_str = ', '.join(plumbing_list)
        if plumbing is not None:
            if not isvalid_slot_value(plumbing, plumbing_list):
                prompt = f"The user was asked to specify the plumbing items [{plumbing_list_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
                bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
                if 'messages' in bedrock_response:
                    message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                    print(f"Extracted message content: {message_content}")
                else:
                    message_content = "I couldn't find any relevant information."

                reply = f"{message_content}\n\nWhich plumbing items do you want covered?\n\n[{plumbing_list_str}]."
                return build_validation_result(False, 'Plumbing', reply)
        else:
            return build_validation_result(
                False,
                'Plumbing',
                f'Please specify the plumbing items you want covered -\n\n[{plumbing_list_str}].'
            )

        # Validate Systems
        systems_list = ['Electrical', 'Fans (Attic, Exhaust, Ceiling, Whole House)', 'Garage Door Opener', 'Garage Door Springs, Hinges, and Transmitters', 'Central Vacuum System', 'All', 'None']
        systems_list_str = ', '.join(systems_list)
        if systems is not None:
            if not isvalid_slot_value(systems, systems_list):
                prompt = f"The user was asked to specify the systems [{systems_list_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
                bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
                if 'messages' in bedrock_response:
                    message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                    print(f"Extracted message content: {message_content}")
                else:
                    message_content = "I couldn't find any relevant information."

                reply = f"{message_content}\n\nWhich systems do you want covered?\n\n[{systems_list_str}]."
                return build_validation_result(False, 'Systems', reply)
        else:
            return build_validation_result(
                False,
                'Systems',
                f'Please specify the systems you want covered -\n\n[{systems_list_str}].'
            )

        # Validate HVAC
        hvac_list = ['Ductwork', 'Heating', 'Refrigerant', 'Air Conditioning', 'Mini-split Ductless Systems', 'Registers, Grills, Filters', 'Window AC Units', 'All', 'None']
        hvac_list_str = ', '.join(hvac_list)
        if hvac is not None:
            if not isvalid_slot_value(hvac, hvac_list):
                prompt = f"The user was asked to specify the HVAC items [{hvac_list_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
                bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
                if 'messages' in bedrock_response:
                    message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                    print(f"Extracted message content: {message_content}")
                else:
                    message_content = "I couldn't find any relevant information."

                reply = f"{message_content}\n\nWhich HVAC items do you want covered?\n\n[{hvac_list_str}]."
                return build_validation_result(False, 'HVAC', reply)
        else:
            return build_validation_result(
                False,
                'HVAC',
                f'Please specify the HVAC items you want covered -\n\n[{hvac_list_str}].'
            )

        # Validate AdditionalLimits
        additional_limits_list = ['Concrete Encasement', 'HVAC Lifting Equipment', 'Improper Installations/Modifications', 'Permits and Code Violations', 'Refrigerant Recapture', 'Reclaim', 'Disposal', 'All', 'None']
        additional_limits_list_str = ', '.join(additional_limits_list)
        if additional_limits is not None:
            if not isvalid_slot_value(additional_limits, additional_limits_list):
                prompt = f"The user was asked to specify the additional limits [{additional_limits_list_str}] as part of a home warranty insurance quote request and this was their response: {intent_request['inputTranscript']}"
                bedrock_response = bedrock_query_knowledge_base(intent_request, prompt)
                if 'messages' in bedrock_response:
                    message_content = next((message['content'] for message in bedrock_response['messages'] if message['contentType'] == 'PlainText'), "")
                    print(f"Extracted message content: {message_content}")
                else:
                    message_content = "I couldn't find any relevant information."

                reply = f"{message_content}\n\nWhich additional limits do you want covered?\n\n[{additional_limits_list_str}]."
                return build_validation_result(False, 'AdditionalLimits', reply)
        else:
            return build_validation_result(
                False,
                'AdditionalLimits',
                f'Please specify the additional limits you want covered -\n\n[{additional_limits_list_str}].'
            )

        # Validate PolicyStartDate
        if policy_start_date is None:
            return build_validation_result(
                False,
                'PolicyStartDate',
                'When would you like your policy to start?'
            )


        # Validate PolicyEndDate
        if policy_end_date is None:
            return build_validation_result(
                False,
                'PolicyEndDate',
                'When would you like your policy to end?'
            )

        '''
        start_date = dateparser.parse(policy_start_date)
        end_date = dateparser.parse(policy_end_date)
        print(f"start_date: {start_date}; end_date: {end_date}")

        # Ensure PolicyEndDate is after PolicyStartDate and at least 30 days apart
        if end_date <= start_date:
            return build_validation_result(
                False,
                'PolicyEndDate',
                'Policy end date must be after the start date. Please enter a valid end date.'
            )
        elif (end_date - start_date).days < 30:
            return build_validation_result(
                False,
                'PolicyEndDate',
                'Policy end date must be at least 30 days after the start date. Please enter a valid end date.'
            )
        '''

    return {'isValid': True}


def generate_home_warranty_quote(intent_request):
    """
    Performs dialog management and fulfillment for completing an insurance quote request.
    """
    slots = intent_request['sessionState']['intent']['slots']
    username = try_ex(slots['UserName'])
    plan_type = try_ex(slots['PlanType'])

    confirmation_status = intent_request['sessionState']['intent']['confirmationState']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    if intent_request['invocationSource'] == 'DialogCodeHook':    

        validation_result = validate_home_warranty_quote(intent_request, slots)

        if not validation_result['isValid']:
            slots = intent_request['sessionState']['intent']['slots']
            slots[validation_result['violatedSlot']] = None

            return elicit_slot(
                session_attributes,
                active_contexts,
                intent_request['sessionState']['intent'],
                validation_result['violatedSlot'],
                validation_result['message']
            )

    if username and plan_type:

        # Determine which PDF to fill out based on coverage type
        pdf_template = 'home_warranty_quote_request.pdf'
        fields_to_update = {}

        # Determine if the intent and current slot settings have been denied
        if confirmation_status == 'Denied' or confirmation_status == 'None':
            return delegate(session_attributes, active_contexts, intent, 'How else can I help you?')

        if confirmation_status == 'Confirmed':
            intent['confirmationState']="Confirmed"
            intent['state']="Fulfilled"

        # PDF generation and S3 upload logic
        s3_client.download_file(s3_artifact_bucket, f'agent/assets/{pdf_template}', f'/tmp/{pdf_template}')

        reader = pdfrw.PdfReader(f'/tmp/{pdf_template}')
        acroform = reader.Root.AcroForm

        # Get the fields from the PDF
        fields = reader.Root.AcroForm.Fields

        # Extract and print field names
        field_names = [field['/T'][1:-1] for field in fields if '/T' in field]

        # Loop through the slots to update fields and create fields_to_update dict
        for slot_name, slot_value in slots.items():
            field_name = slot_name.replace('_', ' ')  # Adjust field naming if necessary
            if field_name and slot_value:
                fields_to_update[field_name] = slot_value['value']['interpretedValue']

        # Update PDF fields
        if acroform is not None and '/Fields' in acroform:
            fields = acroform['/Fields']
            for field in fields:
                field_name = field['/T'][1:-1]  # Extract field name without '/'
                if field_name in fields_to_update:
                    field.update(pdfrw.PdfDict(V=fields_to_update[field_name]))

        writer = pdfrw.PdfWriter()
        writer.addpage(reader.pages[0])  # Assuming you are updating the first page

        completed_pdf_path = f'/tmp/{pdf_template.replace(".pdf", "-completed.pdf")}'
        with open(completed_pdf_path, 'wb') as output_stream:
            writer.write(output_stream)
            
        s3_client.upload_file(completed_pdf_path, s3_artifact_bucket, f'agent/assets/{pdf_template.replace(".pdf", "-completed.pdf")}')

        # Create insurance quote doc in S3
        URLs = []
        URLs.append(create_presigned_url(s3_artifact_bucket, f'agent/assets/{pdf_template.replace(".pdf", "-completed.pdf")}', 3600))
        insurance_quote_link = f'Your home warranty quote request is ready! Please follow the link for details: {URLs[0]}'

        # Write insurance quote request data to DynamoDB
        quote_request = {}

        # Loop through the slots to add items to quote_request dict
        for slot_name, slot_value in slots.items():
            if slot_value:
                quote_request[slot_name] = slot_value['value']['interpretedValue']

        # Convert the JSON document to a string
        quote_request_string = json.dumps(quote_request)

        # Write the JSON document to DynamoDB
        insurance_quote_requests_table = dynamodb.Table(home_warranty_quote_requests_table_name)

        response = insurance_quote_requests_table.put_item(
            Item={
                'UserName': username,
                'RequestTimestamp': int(time.time()),
                'quoteRequest': quote_request_string
            }
        )

        print("Home Warranty Quote Request Submitted Successfully")

        return elicit_intent(
            intent_request,
            session_attributes,
            insurance_quote_link
        )


# DEV BREAK


def generate_claim_summary(claims, username, policy_id):
    """
    Generates a summary of the claims data.
    """
    summary = []
    for claim in claims:
        summary.append(
            f"Claim ID: {claim['ClaimId']}, Type: {claim['ClaimType']}, "
            f"Date: {claim['Incident']['Date']}, Amount: ${claim['ClaimAmount']['Total']}, "
            f"Status: {claim['Status']}, Description: {claim['Damage']['Description']}"
        )
    return f"{username}, here is the summary of your claims for policy ID, {policy_id} -\n\n" + "\n".join(summary)


def summarize_claims(intent_request):
    """
    Summarizes claims data from 'claims_table_name' DynamoDB table.
    """
    slots = intent_request['sessionState']['intent']['slots']
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    intent = intent_request['sessionState']['intent']
    active_contexts = {}

    # Retrieve slot values, check session attributes first
    username = try_ex(slots.get('UserName')) or session_attributes.get('UserName')
    policy_id = try_ex(slots.get('PolicyId')) or session_attributes.get('PolicyId')

    print(f"summarize_claims - session_attributes: {session_attributes}; username: {username}; policy_id: {policy_id}")

   # Validate the slot values
    if not policy_id:
        return elicit_slot(session_attributes, active_contexts, intent, 'PolicyId', 'Please provide your policy ID.')

    claims_table = dynamodb.Table(claims_table_name)

    try:
        # Set up the query parameters
        params = {
            'KeyConditionExpression': Key('PolicyId').eq(policy_id)
        }

        # Execute the query and get the result
        response = claims_table.query(**params)

        # Check if any items were returned
        message = "No claims found for the given policy ID."
        if response['Count'] != 0:
            claims = response['Items']
            message = generate_claim_summary(claims, username, policy_id)
            message += " How else can I help you?"
        return elicit_intent(intent_request, session_attributes, message)

    except Exception as e:
        print(f"Error summarizing claims: {e}")
        return str(e)


def generate_session_id():
    return ''.join(random.choices(string.digits, k=5))


def bedrock_agent(intent_request):
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    query = intent_request['inputTranscript']
    print(f"Bedrock Agent query: {query}")
    
    agent_alias_id = "TSTALIASID"
    agent_id = "U8TWVNTXPJ"

    # Check if session_id exists in session_attributes, if not generate a new one
    session_id = session_attributes.get("SessionId")
    if not session_id:
        session_id = generate_session_id()
        session_attributes["SessionId"] = session_id

    try:    
        # Build the parameters for the invoke_agent call
        params = {
            'agentAliasId': agent_alias_id,
            'agentId': agent_id,
            'sessionId': session_id,
            'inputText': query,
            'enableTrace': True
        }
        
        print("### HERE ###")
        # Invoke the agent
        response = agent_runtime_client.invoke_agent(**params)
        print(f"bedrock_agent response: {response}")
        
        # Handle the response (this is a simplification, adjust as needed for your use case)
        for event in response['completion']:
            if 'chunk' in event:
                result_bytes = event['chunk']['bytes']
                result_text = result_bytes.decode('utf-8')
                print(f"Bedrock Agent response: {result_text}\n")

    except Exception as e:
        result_text = f"Error invoking agent: {e}"

    return elicit_intent(intent_request, session_attributes, result_text)


def place_outgoing_call(intent_request):
    # Initialize a boto3 client for Amazon Connect
    connect_client = boto3.client('connect', region_name='us-east-1')  # Replace with your region
    source_phone_number = "+16592752503"
    dest_phone_number = "+16197090050"
    contact_flow_id = "b3d8f450-edaf-414f-84f9-eff47f96e4b3"
    connect_instance_id = "17f44713-b8ff-42cf-aa32-e86750cc6305"
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    print(f"place_outgoing_call session_attributes: {session_attributes}")

    try:
        response = connect_client.start_outbound_voice_contact(
            SourcePhoneNumber=source_phone_number,
            DestinationPhoneNumber=dest_phone_number,
            ContactFlowId=contact_flow_id,
            InstanceId=connect_instance_id,
            Attributes=session_attributes,
        )
        print(f"Call initiated successfully: {response}")
    except Exception as e:
        print(f"Failed to place call: {e}")


'''def invoke_agent(prompt, session_id):
    """
    Invokes Amazon Bedrock-powered LangChain conversational agent with 'prompt' input.
    """
    chat = Chat({'Human': prompt}, session_id)
    llm = Bedrock(client=bedrock_client, model_id="anthropic.claude-v2:1", region_name=os.environ['AWS_REGION']) # anthropic.claude-instant-v1 / anthropic.claude-3-sonnet-20240229-v1:0
    llm.model_kwargs = {'max_tokens_to_sample': 350}
    lex_agent = HomeWarrantyAgent(llm, chat.memory)
    
    message = lex_agent.run(input=prompt)

    # Summarize response and save in memory
    formatted_prompt = "\n\nHuman: " + "Summarize the following within 50 words: " + message + " \n\nAssistant:"
    conversation = ConversationChain(llm=llm)
    ai_response_recap = conversation.predict(input=formatted_prompt)
    chat.set_memory({'Assistant': ai_response_recap}, session_id)

    return message

def genai_intent(intent_request):
    """
    Performs dialog management and fulfillment for user utterances that do not match defined intents (e.g., FallbackIntent).
    Sends user utterance to the 'invoke_agent' method call.
    """
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    session_id = intent_request['sessionId']
    
    if intent_request['invocationSource'] == 'DialogCodeHook':
        prompt = intent_request['inputTranscript']
        output = invoke_agent(prompt, session_id)
        print(f"Home Warranty Agent response: {output}")

    return elicit_intent(intent_request, session_attributes, output)'''

def bedrock_query_knowledge_base(intent_request, prompt):
    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    session_id = session_attributes.get("sessionId")
    print(f"Knowledge Base query: {prompt}")

    prompt_template = """\n\nHuman: Imagine you are First American Home Warranty's AI Assistant. You respond quickly and friendly to questions from a user, providing both an answer and the sources used to find that answer.
    Format your response for enhanced human readability, using emojis where appropriate. At the end of your response, include the relevant sources if information from specific sources was used in your response. Use the following format for each of the sources used: [Source #: Source Title - Source Link].
    Using the following context, answer the following question to the best of your ability. Do not include information that is not relevant to the question, and only provide information based on the context provided without making assumptions. 
    Do not mention the context and treat each response like it is a continuation of the conversation, so you do not need to greet the user each time.
    Here is the relevant information in numbered order from our knowledge base: $search_results$
    Current time: $current_time$
    User query: $query$\n\nAssistant: """

    model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
    kb_id = "D3IJJEQWJ5"
    filter_attribute = None

    payload = {
        "input": {
            "text": prompt
        },
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": model_arn,
                "generationConfiguration": {
                    "promptTemplate": {
                        "textPromptTemplate": prompt_template
                    }
                },
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                    }
                }
            }
        }
    }

    if filter_attribute is not None:
        print(f"filter_attribute: {filter_attribute}")
        payload["retrieveAndGenerateConfiguration"]["knowledgeBaseConfiguration"]["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = {
            "equals": {
                "key": "exposure",
                "value": filter_attribute
            }
        }

    if session_id is not None:
        print(f"session_id: {session_id}")
        payload["sessionId"] = session_id

    try:
        response = agent_runtime_client.retrieve_and_generate(**payload)
        print(f"BEDROCK RESPONSE: {response}")

        # Extract sessionId from the response and store it in session attributes
        new_session_id = response.get('sessionId')
        if new_session_id:
            session_attributes["sessionId"] = new_session_id

        generated_text = response.get('output', {}).get('text', "")
        citations = response.get('citations', [])
        
        # Extract and append URLs from retrievedReferences
        sources = []
        for citation in citations:
            for ref in citation.get('retrievedReferences', []):
                location = ref.get('location', {})
                url = None
                if 'webLocation' in location:
                    url = location['webLocation'].get('url')
                elif 's3Location' in location:
                    url = location['s3Location'].get('uri')
                elif 'confluenceLocation' in location:
                    url = location['confluenceLocation'].get('url')
                elif 'salesforceLocation' in location:
                    url = location['salesforceLocation'].get('url')
                elif 'sharePointLocation' in location:
                    url = location['sharePointLocation'].get('url')

                if url:
                    sources.append(url)

        if sources:
            sources_text = "\n\nSources:\n" + "\n".join(f"- {source}" for source in sources)
            generated_text += sources_text

        if generated_text:
            print(f"Knowledge Base response: {generated_text}\n")
        else:
            generated_text = "No relevant information found in the knowledge base."

        return elicit_intent(intent_request, session_attributes, generated_text)

    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return f"Error querying knowledge base: {e}"

# --- Intents ---

def dispatch(intent_request):
    """
    Routes the incoming request based on intent.
    """
    slots = intent_request['sessionState']['intent']['slots']
    username = slots['UserName'] if 'UserName' in slots else None
    intent_name = intent_request['sessionState']['intent']['name']
    prompt = intent_request['inputTranscript']

    session_attributes = intent_request['sessionState'].get("sessionAttributes") or {}
    contact_attributes = intent_request['requestAttributes'] if 'requestAttributes' in intent_request else {}

    print(f"Dispatch Session Attributes: {session_attributes}")
    print(f"Dispatch Contact Attributes: {contact_attributes}")

    user_name = contact_attributes.get('UserName', session_attributes.get('UserName', ''))
    policy_id = contact_attributes.get('PolicyId', session_attributes.get('PolicyId', ''))

    print(f"UserName: {user_name}")
    print(f"PolicyId: {policy_id}")

    # Log the event for debugging purposes
    # print("Received event: " + json.dumps(intent_request, indent=2))

    if intent_name == 'VerifyIdentity':
        return verify_identity(intent_request)
    elif intent_name == 'HomeWarrantyQuote':
        return generate_home_warranty_quote(intent_request)
    elif intent_name == 'SummarizeClaims':
        return summarize_claims(intent_request)
    elif intent_name == 'BedrockAgent':
        return bedrock_agent(intent_request)
    elif intent_name == 'OutboundContact':
        return place_outgoing_call(intent_request)
    else:
        # return genai_intent(intent_request)
        return bedrock_query_knowledge_base(intent_request, prompt)

    raise Exception('Intent with name ' + intent_name + ' not supported')
        
# --- Main handler ---

def handler(event, context):
    """
    Invoked when the user provides an utterance that maps to a Lex bot intent.
    The JSON body of the user request is provided in the event slot.
    """
    os.environ['TZ'] = 'America/New_York'
    time.tzset()

    contact_id = event.get('contactId')
    instance_id = event.get('instanceId')
    attributes = event.get('attributes', {})
    parameters = event.get('parameters', {})

    # Example of accessing a specific parameter
    print(f"contact_id: {contact_id}; parameters: {parameters}")

    return dispatch(event)

