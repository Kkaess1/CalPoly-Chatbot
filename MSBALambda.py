import json
import boto3
import os
import logging
import re
from botocore.config import Config

# --- Initialization & Logging ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configure clients with retry logic for Bedrock
config = Config(
    retries={'max_attempts': 3, 'mode': 'adaptive'},
    connect_timeout=5,
    read_timeout=60
)

bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=os.environ['REGION'], config=config)

# Environment Variables
model_id_raw = os.environ['MODEL_ID']
MODEL_ID = model_id_raw.split('/')[-1] if '/' in model_id_raw else model_id_raw
KNOWLEDGE_BASE_ID = os.environ['KNOWLEDGE_BASE_ID']
ALLOWED_ORIGIN = os.environ['ALLOWED_ORIGIN']

# --- Helper Functions ---

def clean_urls(text: str) -> str:
    """
    Removes trailing punctuation from URLs (., !, ?, ), ])
    without affecting valid URL characters.
    """
    # Regex to find standard http/https URLs
    url_pattern = r'(https?://[^\s]+)'

    def strip_trailing_punct(match):
        url = match.group(0)
        # Rstrips characters often accidentally included by LLMs at the end of a sentence
        return url.rstrip('.,!?)]}')

    return re.sub(url_pattern, strip_trailing_punct, text)

# --- Main Lambda Handler ---

def handler(event, context):
    logger.info("Received request")

    # 1. Handle CORS & Method Extraction
    method = (
        event.get('httpMethod') or
        event.get('requestContext', {}).get('http', {}).get('method') or
        event.get('requestContext', {}).get('httpMethod')
    )

    # Standard Headers for Response
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Content-Type': 'application/json'
    }

    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}

    try:
        # 2. Parse Request Body
        body = json.loads(event.get('body', '{}'))
        user_message = body.get('message', '').strip()

        # Default response for empty input
        if not user_message:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'answer': 'Hi! How can I help you with the Cal Poly MSBA program today?'
                })
            }

        # 3. Construct the RAG Prompt
        prompt = f"""
You are PolyBot, a friendly and helpful assistant for Cal Poly's Master of Science in Business Analytics (MSBA) program.
You must answer questions using ONLY verified information contained in the knowledge base.
Do not invent, assume, infer, or guess any information.

If a question is unclear, unrelated to the MSBA program, or not covered by the knowledge base:
- Politely say that you do not have that information
- Suggest a relevant topic, or direct the user to an appropriate official resource

Response Guidelines:
- Keep answers conversational, clear, and concise (about 100 words or fewer)
- Do not display search queries, retrieval actions, or internal reasoning
- Use natural, helpful language

Links and URLs:
- Only include URLs that appear in the knowledge base
- Never fabricate or modify URLs
- Do NOT add punctuation (such as periods, exclamation points, or brackets) immediately after URLs
- Official Contact Form: https://forms.office.com/pages/responsepage.aspx?id=2wING578lUSVNx03nMoq5_unPySOHhVKs6D8naanJd5UMkswTFVONVBRME02WllTWjdSSjJCU0o2OS4u&route=shorturl
- FAQs: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics/faqs
- Information Sessions: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics#information-sessions

Program-Specific Rules:
- If asked about prerequisite courses: Explain that students must have completed one college-level calculus course and two college-level statistics courses.

Question: {user_message}
"""

        # 4. Query Bedrock Knowledge Base
        response = bedrock_agent.retrieve_and_generate(
            input={'text': prompt.strip()},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                    'modelArn': MODEL_ID
                }
            }
        )

        # 5. Extract and Clean Answer
        raw_answer = response['output']['text']
        final_answer = clean_urls(raw_answer)

        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({'answer': final_answer})
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Unable to process your request. Please try again.'
            })
        }
