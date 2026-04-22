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

# Environment Variables — MODEL_ID must be the full ARN
MODEL_ID = os.environ['MODEL_ID']
KNOWLEDGE_BASE_ID = os.environ['KNOWLEDGE_BASE_ID']
ALLOWED_ORIGIN = os.environ['ALLOWED_ORIGIN']

# --- Helper Functions ---

def clean_urls(text: str) -> str:
    """
    Removes trailing punctuation from URLs (., !, ?, ), ])
    without affecting valid URL characters.
    """
    url_pattern = r'(https?://[^\s]+)'

    def strip_trailing_punct(match):
        url = match.group(0)
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

        logger.info(f"User message: {user_message}")
        logger.info(f"Using MODEL_ID: {MODEL_ID}")
        logger.info(f"Using KNOWLEDGE_BASE_ID: {KNOWLEDGE_BASE_ID}")

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
Your job is to answer student questions using only the retrieved knowledge base content provided to you.

<instructions>
1. Answer using only retrieved content — do not invent, assume, or infer information
2. Keep responses conversational, clear, and concise (100 words or fewer, expanding only when necessary)
3. Only include URLs that appear exactly in the retrieved content — never fabricate or modify them
4. Do not add punctuation immediately after any URL
5. If a question is unclear, off-topic, or not covered by retrieved content, politely say so and point the user to a relevant resource below
6. You remember the conversation history — use prior questions and answers to give contextually aware responses

Useful resources (only cite if relevant):
- Contact Form: https://forms.office.com/pages/responsepage.aspx?id=2wING578lUSVNx03nMoq5_unPySOHhVKs6D8naanJd5UMkswTFVONVBRME02WllTWjdSSjJCU0o2OS4u&route=shorturl
- FAQs: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics/faqs
- Information Sessions: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics#information-sessions

For prerequisite questions: students must have completed one college-level calculus course and two college-level statistics courses.
</instructions>

<examples>
Q: What are the prerequisites for the MSBA program?
A: To apply, you'll need one college-level calculus course and two college-level statistics courses completed. For more details, check the FAQs: https://orfalea.calpoly.edu/graduate-programs/ms-business-analytics/faqs

Q: Can you tell me about parking on campus?
A: That falls outside what I can help with here! For questions unrelated to the MSBA program, I'd recommend reaching out directly: https://forms.office.com/pages/responsepage.aspx?id=2wING578lUSVNx03nMoq5_unPySOHhVKs6D8naanJd5UMkswTFVONVBRME02WllTWjdSSjJCU0o2OS4u&route=shorturl
</examples>

Question: {user_message}
Answer:
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

        logger.info(f"Successfully generated answer")

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