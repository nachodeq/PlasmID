from django.shortcuts import render, redirect
from pymongo import MongoClient
from datetime import datetime
import json
import logging
import csv
from bson import ObjectId
from django.http import JsonResponse, HttpResponse
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain  # Corrected import
import re
from django.views.decorators.csrf import csrf_protect

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

# Configure logging handlers (console and file)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('django_app.log')
file_handler.setLevel(logging.DEBUG)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Constants
DISPLAY_LIMIT = 10          # Limit query results displayed to 10
MAX_FIELD_LENGTH = 100      # Maximum characters per field in results
MAX_QUERY_LENGTH = 500      # Maximum characters for display-only query fields

# --- MongoDB Connection Utility ---
def get_mongo_client():
    """
    Establishes and returns a MongoDB client.
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")  # Adjust URI as needed
        logger.info("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def get_db():
    """
    Returns the MongoDB database instance.
    """
    client = get_mongo_client()
    db = client['PruebaTFMallplasmids']
    logger.debug("Accessed MongoDB database: PruebaTFMallplasmids")
    return db

# --- Helper Functions ---

def database_schema(request):
    """
    Renders the Database Schema page displaying the MongoDB schema and explanatory text.
    """
    schema = {
        "genes": {
            "_id": "ObjectId",
            "plasmid_id": "ObjectId",
            "locus": "String",
            "gene_id": "String",
            "gene_name": "String",
            "product": "String",
            "start": "Int32",
            "stop": "Int32",
            "strand": "String",
            "contig": "String",
            "nt_sequence": "String",
            "aa_sequence": "String",
            "antibiotic_resistance": "Boolean",
            "resistance_info": {
                "gene_name": "String",
                "resistance_to": ["String"],
                "identity": "Double",
                "coverage": "Double",
                "alignment_length": "String"
            }
        },
        "plasmids": {
            "_id": "ObjectId",
            "plasmid_id": "String",
            "sequence": "String",
            "sequence_length": "Int32",
            "mobility": "String",
            "replicon_type": "String",
            "environment_id": "ObjectId",
            "host_id": "ObjectId",
            "assembly_status": "String",
            "assembly_accession": "String"
        },
        "hosts": {
            "_id": "ObjectId",
            "genus": "String",
            "species": "String",
            "family": "String",
            "environment_ids": ["ObjectId"]
        },
        "environments": {
            "_id": "ObjectId",
            "name": "String"
        }
    }

    explanatory_text = (
        "This is the database schema (i.e., what is in the database and how it is related). "
        "This may come in handy if you're trying to make complex queries. Feel free to use it as a prompt "
        "if you are using a Large Language Model (LLM) to help you make queries."
    )

    context = {
        'schema': schema,
        'explanatory_text': explanatory_text
    }

    logger.debug("Rendering database_schema.html with schema and explanatory text.")
    return render(request, 'database_schema.html', context)


def convert_objectids(data):
    """
    Recursively traverse the data and convert ObjectId instances to strings.
    """
    if isinstance(data, dict):
        return {k: convert_objectids(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectids(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    else:
        return data 

def truncate_string_fields(data, max_length=MAX_FIELD_LENGTH):
    """
    Recursively traverse the data and truncate any string fields to max_length.
    """
    if isinstance(data, dict):
        return {k: truncate_string_fields(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_string_fields(item, max_length) for item in data]
    elif isinstance(data, str):
        return data[:max_length]
    else:
        return data

def replace_regex_literals(json_string):
    """
    Replaces regex literals in the form /pattern/flags with JSON-compatible
    {"$regex": "pattern", "$options": "flags"} objects.

    Args:
        json_string (str): The JSON string containing regex literals.

    Returns:
        str: The JSON string with regex literals replaced.
    """
    # Updated regex pattern without the trailing slash
    regex_literal_pattern = r'(":\s*)/([^/]+)/([a-z]*)'

    def regex_replacer(match):
        prefix = match.group(1)        # The part before the regex literal (e.g., "field": )
        pattern = match.group(2)       # The regex pattern (e.g., ctx-m-15)
        flags = match.group(3)         # The regex flags (e.g., i)

        # Map JavaScript regex flags to MongoDB options
        flag_mapping = {
            'i': 'i',  # Case-insensitive
            'm': 'm',  # Multiline
            's': 's',  # Dotall
            'x': 'x',  # Extended
            # Add more mappings if necessary
        }

        # Extract relevant options based on flags
        options = ''.join([flag_mapping.get(flag, '') for flag in flags])

        # Construct the replacement JSON object
        replacement = {
            "$regex": pattern,
            "$options": options
        }

        # Convert the replacement object to a JSON string
        replacement_str = json.dumps(replacement)

        return f'{prefix}{replacement_str}'

    # Replace all occurrences of regex literals with JSON-compatible objects
    cleaned_json_string = re.sub(regex_literal_pattern, regex_replacer, json_string)

    return cleaned_json_string

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    :param d: Dictionary to flatten
    :param parent_key: String of parent key
    :param sep: Separator between keys
    :return: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert list to comma-separated string
            list_items = []
            for item in v:
                if isinstance(item, dict):
                    list_items.append(json.dumps(flatten_dict(item)))
                else:
                    list_items.append(str(item))
            items.append((new_key, ', '.join(list_items)))
        else:
            items.append((new_key, v))
    return dict(items)


# -------------------- Define the Table Schema --------------------
TABLE_SCHEMA = '''
Collections and Their Fields:

1. **environments**
    - `_id`: ObjectId
    - `name`: String

2. **genes**
    - `_id`: ObjectId
    - `aa_sequence`: String
    - `antibiotic_resistance`: Boolean
    - `contig`: String
    - `gene_id`: String
    - `gene_name`: String
    - `locus`: String
    - `nt_sequence`: String
    - `plasmid_id`: ObjectId
    - `product`: String  **this is the field to use when asking for a product**
    - `resistance_info`: 
        - `alignment_length`: String
        - `coverage`: Double
        - `gene_name`: String
        - `identity`: Double
        - `resistance_to`: Array of Strings
    - `start`: Int32
    - `stop`: Int32
    - `strand`: String

3. **hosts**
    - `_id`: ObjectId
    - `environment_ids`: Array of ObjectIds
    - `family`: String
    - `genus`: String
    - `species`: String

4. **plasmids**
    - `_id`: ObjectId
    - `assembly_accession`: String
    - `assembly_status`: String
    - `environment_id`: ObjectId
    - `host_id`: ObjectId
    - `mobility`: String
    - `plasmid_id`: String
    - `replicon_type`: String
    - `sequence`: String
    - `sequence_length`: Int32
'''
# ------------------------------------------------------------------------

# -------------------- Define the Schema Description --------------------
SCHEMA_DESCRIPTION = '''
Please only use the fields specified here using their exact names

Detailed Descriptions of Collections and Their Fields:

1. **environments**
    - `_id`: Unique identifier for the environment document.
    - `name`: Name of the environment.

2. **genes**
    - `_id`: Unique identifier for the gene document.
    - `aa_sequence`: Amino acid sequence of the gene.
    - `antibiotic_resistance`: Indicates if the gene confers antibiotic resistance.
    - `contig`: Contig information.
    - `gene_id`: Identifier of the gene.
    - `gene_name`: Name of the gene.
    - `locus`: Locus information.
    - `nt_sequence`: Nucleotide sequence of the gene.
    - `plasmid_id`: Reference to the associated plasmid (`_id` from the `plasmids` collection).
    - `product`: Product of the gene.
    - `resistance_info`: Contains resistance information such as:
        - `alignment_length`: Length of the alignment.
        - `coverage`: Coverage percentage (Double).
        - `gene_name`: Name of the resistance gene.
        - `identity`: Identity percentage (Double).
        - `resistance_to`: List of antibiotics the gene provides resistance to.
    - `start`: Start position of the gene (Int32).
    - `stop`: Stop position of the gene (Int32).
    - `strand`: Strand information.

3. **hosts**
    - `_id`: Unique identifier for the host document.
    - `environment_ids`: List of environment IDs associated with the host.
    - `family`: Family classification.
    - `genus`: Genus classification.
    - `species`: Species classification.

4. **plasmids**
    - `_id`: Unique identifier for the plasmid document.
    - `assembly_accession`: Assembly accession number.
    - `assembly_status`: Status of the assembly.
    - `environment_id`: Reference to the associated environment (`_id` from the `environments` collection).
    - `host_id`: Reference to the associated host (`_id` from the `hosts` collection).
    - `mobility`: Mobility information.
    - `plasmid_id`: Identifier of the plasmid.
    - `replicon_type`: Type of replicon.
    - `sequence`: DNA sequence of the plasmid.
    - `sequence_length`: Length of the plasmid sequence (Int32).
'''
# ------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # Example 1: Retrieve specific fields with $project
    {
        "input": "List the gene names and resistance information for genes that confer antibiotic resistance and have a coverage greater than 90%.",
        "output": {
            "collection": "genes",
            "pipeline": [
                {
                    "$match": {
                        "antibiotic_resistance": True,
                        "resistance_info.coverage": {"$gt": 90}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "gene_name": 1,
                        "resistance_info": 1
                    }
                }
            ]
        }
    },
    # Example 2: Retrieve all fields without $project
    {
        "input": "Find all genes associated with plasmids having a sequence length greater than 5000.",
        "output": {
            "collection": "genes",
            "pipeline": [
                {
                    "$lookup": {
                        "from": "plasmids",
                        "localField": "plasmid_id",
                        "foreignField": "_id",
                        "as": "plasmid_info"
                    }
                },
                {
                    "$match": {
                        "plasmid_info.sequence_length": {"$gt": 5000}
                    }
                }
                # No $project stage since all fields are required
            ]
        }
    },
    # Example 3: Retrieve specific fields with $project
    {
        "input": "List genes with a start position before 1000 and stop position after 5000.",
        "output": {
            "collection": "genes",
            "pipeline": [
                {
                    "$match": {
                        "start": {"$lt": 1000},
                        "stop": {"$gt": 5000}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "gene_name": 1,
                        "start": 1,
                        "stop": 1
                    }
                }
            ]
        }
    },
    # Example 4: Retrieve specific fields with $project
    {
        "input": "Retrieve all genes that provide resistance to 'Ampicillin' with an identity greater than 98%.",
        "output": {
            "collection": "genes",
            "pipeline": [
                {
                    "$match": {
                        "resistance_info.resistance_to": "Ampicillin",
                        "resistance_info.identity": {"$gt": 98}
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "gene_name": 1,
                        "resistance_info": 1
                    }
                }
            ]
        }
    },
    # Example 5: Retrieve all fields without $project
    {
        "input": "Find all hosts that belong to the genus 'Escherichia' and are associated with the environment named 'Hospital'.",
        "output": {
            "collection": "hosts",
            "pipeline": [
                {
                    "$lookup": {
                        "from": "environments",
                        "localField": "environment_ids",
                        "foreignField": "_id",
                        "as": "environment_info"
                    }
                },
                {
                    "$match": {
                        "genus": "Escherichia",
                        "environment_info.name": "Hospital"
                    }
                }
                # No $project stage since all fields are required
            ]
        }
    },
    # Example 6: Retrieve specific fields with $project
    {
        "input": "List all plasmids with mobility type 'Conjugative' and associated with hosts from the family 'Enterobacteriaceae'.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$lookup": {
                        "from": "hosts",
                        "localField": "host_id",
                        "foreignField": "_id",
                        "as": "host_info"
                    }
                },
                {
                    "$match": {
                        "mobility": "Conjugative",
                        "host_info.family": "Enterobacteriaceae"
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "assembly_accession": 1,
                        "mobility": 1,
                        "host_info.family": 1
                    }
                }
            ]
        }
    },
    # Example 7: Retrieve all fields without $project
    {
        "input": "Find all plasmids between 10000 and 30000 bases.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$match": {
                        "sequence_length": {
                            "$gt": 10000,
                            "$lt": 30000
                        }
                    }
                }
                # No $project stage; all fields will be returned
            ]
        }
    },
    # Example 8: Retrieve specific fields with $project
    {
        "input": "List the assembly accession and sequence length for plasmids between 10000 and 30000 bases.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$match": {
                        "sequence_length": {
                            "$gt": 10000,
                            "$lt": 30000
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "assembly_accession": 1,
                        "sequence_length": 1
                    }
                }
            ]
        }
    },
    # Example 9: Retrieve all fields without $project
    {
        "input": "Retrieve all plasmids that contain a replicon type of 'IncQ' and are found in Escherichia hosts.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$lookup": {
                        "from": "hosts",
                        "localField": "host_id",
                        "foreignField": "_id",
                        "as": "host_info"
                    }
                },
                {
                    "$match": {
                        "replicon_type": { "$regex": "IncQ", "$options": "i" },
                        "host_info.genus": "Escherichia"
                    }
                }
                # No $project stage since all fields are required
            ]
        }
    },
    # Example 11: Retrieve genes with specific product patterns
    {
        "input": "Find all genes whose product includes 'CTX-M-15' regardless of case.",
        "output": {
            "collection": "genes",
            "pipeline": [
                {
                    "$match": {
                        "product": {
                            "$regex": "CTX-M-15",
                            "$options": "i"
                        }
                    }
                }
            ]
        }
    },
    # Example 12: Retrieve plasmids along with specific gene products
    {
        "input": "List all plasmids that are associated with genes producing 'CTX-M-15' proteins, showing only plasmid_id and gene product.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$lookup": {
                        "from": "genes",
                        "localField": "_id",
                        "foreignField": "plasmid_id",
                        "as": "genes"
                    }
                },
                {
                    "$unwind": "$genes"
                },
                {
                    "$match": {
                        "genes.product": {
                            "$regex": "CTX-M-15",
                            "$options": "i"
                        }
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "plasmid_id": 1,
                        "genes.product": 1
                    }
                }
            ]
        }
    },
    # Example 10: New Few-Shot Example Including User's Query
    {
        "input": "Find plasmids with 'non-mobilizable' mobility and 'CTX-M-15' as a gene product.",
        "output": {
            "collection": "plasmids",
            "pipeline": [
                {
                    "$match": {
                        "mobility": "non-mobilizable"
                    }
                },
                {
                    "$lookup": {
                        "from": "genes",
                        "localField": "_id",
                        "foreignField": "plasmid_id",
                        "as": "genes"
                    }
                },
                {
                    "$unwind": "$genes"
                },
                {
                    "$match": {
                        "genes.product": {
                            "$regex": "CTX-M-15",
                            "$options": "i"
                        }
                    }
                }
            ]
        }
    }
]



def load_additional_examples(file_path):
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data_list = json.loads(content)

    for data in data_list:
        examples.append({
            'input': data['input'],
            'output': data['output']
        })
    return examples


# Assuming your dataset is in 'fine_tune_data.jsonl'
additional_examples = load_additional_examples('/home/nacho/TFM/mongodb/queriesfortraining/few_shot_examples.json')

# Combine with existing examples
FEW_SHOT_EXAMPLES.extend(additional_examples)
# ------------------------------------------------------------------------


# -------------------- Construct the Prompt Template --------------------
# Convert the few-shot examples to a formatted string with escaped curly braces
few_shot_examples = "\n".join([
    f"Input: {example['input']}\nOutput:\n```json\n{json.dumps(example['output'], indent=4).replace('{', '{{').replace('}', '}}')}\n```\n"
    for example in FEW_SHOT_EXAMPLES
])

prompt_template_for_creating_query =prompt_template_for_creating_query = f"""
You are an expert in crafting NoSQL queries for MongoDB with 10 years of experience.
I will provide you with the table schema and schema description.
You must only invoke fields provided in the table schema.
Your task is to read the user question and create a MongoDB aggregation pipeline accordingly.
It is very important that you dont make up any fields and only use the ones in the schema, such as product, dont invent gene_product or product_id
**It is strictly forbidden to use regex literals like /pattern/flags/. Instead, always use "$regex": "pattern", "$options": "flags".**

**Instructions:**

1. **Identify the Correct Collection:**
   - Determine which collection to query based on the user's natural language question.

2. **Use Exact Field Names:**
   - Ensure all field names match exactly as defined in the schema. Do not invent or alter field names. Field names are case-sensitive.

3. **Proper JSON Syntax:**
   - All field names and string values must be enclosed in double quotes.
   - Avoid trailing commas after the last item in objects or arrays.

4. **Reference Nested Fields Correctly:**
   - Use dot notation for nested fields as per the schema.


5. **Include Only Necessary Stages:**
   - Use the `$project` stage only when specific fields need to be included or excluded. If all fields are required, omit the `$project` stage.

6. **Output Format:**
   - Return the result as a JSON object with only two keys: `"collection"` and `"pipeline"`.
   - Enclose the JSON object within triple backticks with `json` specified.
   - Do not include any additional text, explanations, comments, or examples.

**Important:**

- **Do Not** include any additional text, explanations, comments, or examples.
- **Ensure** that the JSON object is enclosed within ```json code fences without any leading or trailing whitespace.
- **If** the user asks you to look for certain patterns, such as a replicon type, use a regular expression to match any similar, case-insensitive occurrences that may be part of another word.


Table Schema:
{TABLE_SCHEMA}

Schema Description:
{SCHEMA_DESCRIPTION}

Here are some examples:
{few_shot_examples}

### User Question

Note:

- Read the user's question carefully and determine the correct collection to query.
- Create a MongoDB aggregation pipeline that answers the user's question.
- **RETURN ONLY** the collection name and the MongoDB aggregation pipeline as a JSON object with the keys "collection" and "pipeline".
- **DO NOT** include the examples, comments, explanations, or any additional text.
- **Ensure the JSON object is enclosed within ```json code fences without any leading or trailing whitespace.**
- ** If the user asks you to look for certain pattern, such as a replicon type, look for it in a regexp for matching anything similar, case-insensitive and can be contained in other word**

Input: {{user_question}}
"""



# ------------------------------------------------------------------------

# -------------------- Initialize the Prompt and LLMChain --------------------
query_creation_prompt = PromptTemplate(
    input_variables=["user_question"],
    template=prompt_template_for_creating_query,
)

# Initialize the Language Model
try:
    llm = OllamaLLM(model="codellama", temperature=0.2)  # Ensure 'codellama' is the correct model name
    logger.info("Initialized OllamaLLM with CodeLlama model successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OllamaLLM: {e}")
    print(f"Failed to initialize OllamaLLM: {e}")
    exit(1)

# Initialize the LLMChain
try:
    llmchain = LLMChain(llm=llm, prompt=query_creation_prompt, verbose=True)
    logger.info("Initialized LLMChain successfully.")
except Exception as e:
    logger.error(f"Failed to initialize LLMChain: {e}")
    print(f"Failed to initialize LLMChain: {e}")
    exit(1)

# ------------------------------------------------------------------------

# -------------------- Define Helper Functions --------------------

def clean_json_string(json_string):
    """
    Cleans the JSON string by removing code fences, trailing commas, comments, and unnecessary whitespace.

    Args:
        json_string (str): The raw JSON string from the model.

    Returns:
        str: The cleaned JSON string.
    """
    # Remove any code fences (```json and ```)
    json_string = re.sub(r'```json\s*', '', json_string)
    json_string = re.sub(r'```', '', json_string)

    # Remove trailing commas in objects and arrays
    json_string = re.sub(r',\s*(\}|\])', r'\1', json_string)

    # Remove comments (if any)
    json_string = re.sub(r'//.*', '', json_string)

    # Remove unnecessary whitespace
    json_string = json_string.strip()

    return json_string


def strip_keys(obj):
    """
    Recursively strips whitespace and surrounding quotes from all keys in the JSON object.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            # Remove leading/trailing whitespace and surrounding quotes
            new_key = key.strip().strip('"').strip("'")
            new_obj[new_key] = strip_keys(value)
        return new_obj
    elif isinstance(obj, list):
        return [strip_keys(item) for item in obj]
    else:
        return obj


def generate_pipeline(user_question):
    """
    Generates a MongoDB aggregation pipeline based on the user's natural language question.

    Parameters:
        user_question (str): The natural language question.

    Returns:
        dict: A dictionary containing the 'collection' and 'pipeline' as returned by the model.
    """
    if not llmchain:
        logger.error("LLMChain is not initialized.")
        return None

    response = None  # Initialize response to avoid UnboundLocalError
    try:
        # Send the user question to the LLMChain
        response = llmchain.run({"user_question": user_question})

        # Log the raw response for debugging
        logger.debug("Raw response from model:")
        logger.debug(response)

        # Output the raw response to the console (for debugging)
        print("Raw response from model:", response)

        # Use regex to extract JSON object within code fences
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_string = json_match.group(1)
            logger.debug("Extracted JSON from code fences.")
        else:
            # If no code fences, assume entire response is JSON
            json_string = response.strip()
            logger.debug("Assuming entire response is JSON.")

        # Clean the JSON string
        cleaned_response = clean_json_string(json_string)
        logger.debug(f"Cleaned JSON response: {cleaned_response}")

        # Replace regex literals with JSON-compatible syntax
        cleaned_response = replace_regex_literals(cleaned_response)
        logger.debug(f"JSON after regex replacement: {cleaned_response}")

        # Parse the JSON response
        output = json.loads(cleaned_response)
        logger.debug(f"Parsed output: {output}")

        # Validate the output
        if 'collection' in output and 'pipeline' in output:
            # Ensure the pipeline is a flat list of objects
            if isinstance(output['pipeline'], list):
                # Check if the first element is a list (nested)
                if len(output['pipeline']) > 0 and isinstance(output['pipeline'][0], list):
                    # Flatten the pipeline
                    flattened_pipeline = [stage for sublist in output['pipeline'] for stage in sublist]
                    logger.debug("Flattened the nested pipeline.")
                    output['pipeline'] = flattened_pipeline
                else:
                    logger.debug("Pipeline is already a flat list.")
            else:
                logger.error("Pipeline is not a list.")
                return None

            # Optional: Validate that each stage is a dict
            if all(isinstance(stage, dict) for stage in output['pipeline']):
                logger.debug("Pipeline validation successful.")
            else:
                logger.error("Pipeline validation failed. Each stage must be a dictionary.")
                return None

            logger.debug(f"Final pipeline: {output['pipeline']}")
            return output
        else:
            missing_keys = {'collection', 'pipeline'} - set(output.keys())
            logger.error(f"Invalid output format. Missing keys: {missing_keys}")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        if response:
            logger.error(f"Response from model: {response}")
        return None
    except Exception as e:
        logger.error(f"Error generating the pipeline: {e}")
        if response:
            logger.error(f"Response from model: {response}")
        return None




# ------------------------------------------------------------------------

# -------------------- Define Views --------------------

@csrf_protect  # Use this decorator to enable CSRF protection
def natural_language_query(request):
    if request.method == 'POST':
        natural_language_query = request.POST.get('natural_language_query', '').strip()

        logger.debug(f"Received Natural Language Query: {natural_language_query}")

        # Generate the pipeline using the LLM
        output = generate_pipeline(natural_language_query)

        if output:
            json_query_str = json.dumps(output['pipeline'], indent=4)
            target_collection = output['collection']

            context = {
                'natural_language_query': natural_language_query,
                'json_query': json_query_str,
                'target_collection': target_collection,
                'error': '',
                'raw_response': output  # Include the raw output from the LLM
            }

            logger.debug("Rendering natural_language_query.html with generated pipeline.")
            return render(request, 'natural_language_query.html', context)
        else:
            error = "Failed to generate a valid pipeline. Please check your query and try again."
            logger.error(error)
            context = {
                'natural_language_query': natural_language_query,
                'json_query': '',
                'target_collection': '',
                'error': error,
                'raw_response': ''  # Provide a default value to prevent template errors
            }
            return render(request, 'natural_language_query.html', context)
    else:
        # For GET requests, display an empty form
        context = {
            'natural_language_query': '',
            'json_query': '',
            'target_collection': '',
            'error': '',
            'raw_response': ''  # Provide a default value
        }
        return render(request, 'natural_language_query.html', context)



def home_view(request):
    return render(request, 'home.html')

def queries_home(request):
    return render(request, 'queries_home.html')

def premade_queries(request):
    db = get_db()
    queries_collection = db['queries']

    search_query = request.GET.get('search', '').strip()

    if search_query:
        premade_queries_cursor = queries_collection.find({
            'is_premade': True,
            'natural_language_query': {'$regex': search_query, '$options': 'i'}
        })
    else:
        premade_queries_cursor = queries_collection.find({'is_premade': True})

    premade_queries = []
    for q in premade_queries_cursor:
        premade_queries.append({
            'id': str(q['_id']),
            'natural_language_query': (q['natural_language_query'][:MAX_QUERY_LENGTH] 
                                        if 'natural_language_query' in q else ''),
            'json_query': json.dumps(q['json_query'], indent=4),  # Proper formatting with indentation
            'target_collection': q.get('target_collection')  # Include only if set
        })

    context = {
        'queries': premade_queries,
        'search_query': search_query
    }
    logger.debug(f"Rendering premade_queries.html with {len(premade_queries)} queries.")
    return render(request, 'premade_queries.html', context)

def save_queries(request):
    """Handle saving a single query to MongoDB."""
    if request.method == 'POST':
        # Extract fields from the POST request
        natural_language_query = request.POST.get('natural_language_query', '').strip()
        json_query_str = request.POST.get('json_query', '').strip()
        is_premade = True        
        target_collection = request.POST.get('target_collection', '').strip()  # New field

        logger.debug("Processing POST request with save_query flag set.")
        logger.debug(f"Natural Language Query: {natural_language_query}")
        logger.debug(f"JSON Query String: {json_query_str}")
        logger.debug(f"Target Collection: {target_collection}")

        # Validate inputs
        if not natural_language_query or not json_query_str or not target_collection:
            logger.error("Missing natural language query, JSON query string, or target collection.")
            error_message = "Missing required fields: natural_language_query, json_query, target_collection."
            return render(request, 'save_error.html', {'error': error_message})

        # Parse JSON query string
        try:
            json_query = json.loads(json_query_str)
            logger.debug("JSON query successfully parsed.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON query: {e}")
            error_message = "Invalid JSON query format."
            return render(request, 'save_error.html', {'error': error_message})

        # No escaping applied to json_query
        logger.debug(f"JSON query: {json_query}")

        # Prepare the document
        document = {
            "natural_language_query": natural_language_query,
            "json_query": json_query,  # Store as-is without escaping
            "is_premade": is_premade,
            "target_collection": target_collection,  # Store target collection
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Save the query to MongoDB
        try:
            db = get_db()
            queries_collection = db['queries']
            result = queries_collection.update_one(
                {"natural_language_query": natural_language_query},
                {"$set": document},
                upsert=True
            )
            if result.upserted_id:
                logger.info(f"Inserted new query: {natural_language_query}")
                # Render the success template
                return render(request, 'save_success.html', {
                    'natural_language_query': natural_language_query,
                    'action': 'inserted'
                })
            else:
                logger.info(f"Updated existing query: {natural_language_query}")
                # Render the success template
                return render(request, 'save_success.html', {
                    'natural_language_query': natural_language_query,
                    'action': 'updated'
                })
        except Exception as e:
            logger.error(f"Error saving query to MongoDB: {e}")
            error_message = "Database operation failed."
            return render(request, 'save_error.html', {'error': error_message})
    else:
        logger.warning("Invalid request method.")
        error_message = "Invalid request method."
        return render(request, 'save_error.html', {'error': error_message})

def save_success(request):
    return render(request, 'save_success.html')

def execute_query(request):
    if request.method == 'POST':
        json_query_str = request.POST.get('json_query', '').strip()
        natural_language_query = request.POST.get('natural_language_query', '').strip()
        target_collection = request.POST.get('target_collection', '').strip()

        logger.debug(f"Received JSON Query String: {json_query_str}")
        logger.debug(f"Natural Language Query: {natural_language_query}")
        logger.debug(f"Target Collection: {target_collection}")

        try:
            json_query = json.loads(json_query_str)
            logger.debug(f"Parsed JSON Query: {json_query}")
        except json.JSONDecodeError as e:
            error = f'Invalid JSON query format: {str(e)}'
            logger.error(error)
            return render(request, 'query_result.html', {
                'error': error,
                'json_query': json_query_str,
                'natural_language_query': natural_language_query,
                'results': [],
                'target_collection': target_collection
            })

        is_aggregate = isinstance(json_query, list)
        db = get_db()

        try:
            if is_aggregate:
                if not target_collection:
                    error = 'Target collection must be specified for aggregation pipelines.'
                    logger.error(error)
                    return render(request, 'query_result.html', {
                        'error': error,
                        'json_query': json_query_str,
                        'natural_language_query': natural_language_query,
                        'results': [],
                        'target_collection': target_collection
                    })
                collection = db[target_collection]
                logger.debug(f"Executing aggregation pipeline on collection: {target_collection}")
                
                # Remove existing $limit stages if any to control pagination separately
                json_query_cleaned = [stage for stage in json_query if not ('$limit' in stage)]
                
                # Append $limit for display
                json_query_display = json_query_cleaned + [{"$limit": DISPLAY_LIMIT}]
                
                results_cursor = collection.aggregate(json_query_display)
            else:
                if not target_collection:
                    error = 'Target collection must be specified for single-field queries.'
                    logger.error(error)
                    return render(request, 'query_result.html', {
                        'error': error,
                        'json_query': json_query_str,
                        'natural_language_query': natural_language_query,
                        'results': [],
                        'target_collection': target_collection
                    })
                collection = db[target_collection]
                logger.debug(f"Executing find query on collection: {target_collection} with query: {json_query}")
                results_cursor = collection.find(json_query).limit(DISPLAY_LIMIT)

            results = list(results_cursor)

            # Convert ObjectId to string and truncate string fields
            truncated_results = [truncate_string_fields(convert_objectids(result), MAX_FIELD_LENGTH) for result in results]

            # Flatten the results for display
            flattened_results = [flatten_dict(result) for result in truncated_results]

            logger.debug(f"Query returned {len(flattened_results)} results (showing first {DISPLAY_LIMIT}): {flattened_results[:3]}")

            if not flattened_results:
                error = 'No results found.'
            else:
                error = ''

            # Store the full query in session for CSV download
            request.session['current_query'] = {
                'json_query': json_query_str,
                'target_collection': target_collection
            }

        except Exception as e:
            error = f"An error occurred while executing the query: {str(e)}"
            logger.error(f"Query Execution Error: {e}")
            return render(request, 'query_result.html', {
                'error': error,
                'json_query': json_query_str,
                'natural_language_query': natural_language_query,
                'results': [],
                'target_collection': target_collection
            })

        context = {
            'results': flattened_results,
            'json_query': json_query_str,
            'natural_language_query': natural_language_query,
            'error': error,
            'target_collection': target_collection
        }
        return render(request, 'query_result.html', context)

    return redirect('queries_home')

def new_query(request):
    if request.method == 'POST':
        natural_language_query = request.POST.get('natural_language_query', '').strip()
        json_query_str = request.POST.get('json_query', '').strip()
        target_collection = request.POST.get('target_collection', '').strip()

        logger.debug(f"Received JSON Query String: {json_query_str}")
        logger.debug(f"Natural Language Query: {natural_language_query}")
        logger.debug(f"Target Collection: {target_collection}")

        try:
            json_query = json.loads(json_query_str)
            logger.debug(f"Parsed JSON Query: {json_query}")
        except json.JSONDecodeError as e:
            error = f'Invalid JSON query format: {str(e)}'
            logger.error(error)
            return render(request, 'new_query.html', {
                'error': error,
                'natural_language_query': natural_language_query,
                'json_query': json_query_str,
                'results': []
            })

        is_aggregate = isinstance(json_query, list)
        db = get_db()

        try:
            if is_aggregate:
                if not target_collection:
                    error = 'Target collection must be specified for aggregation pipelines.'
                    logger.error(error)
                    return render(request, 'new_query.html', {
                        'error': error,
                        'natural_language_query': natural_language_query,
                        'json_query': json_query_str,
                        'results': []
                    })
                collection = db[target_collection]
                logger.debug(f"Executing aggregation pipeline on collection: {target_collection}")
                
                # Remove existing $limit stages if any to control pagination separately
                json_query_cleaned = [stage for stage in json_query if not ('$limit' in stage)]
                
                # Append $limit for display
                json_query_display = json_query_cleaned + [{"$limit": DISPLAY_LIMIT}]
                
                results_cursor = collection.aggregate(json_query_display)
            else:
                if not target_collection:
                    error = 'Target collection must be specified for single-field queries.'
                    logger.error(error)
                    return render(request, 'new_query.html', {
                        'error': error,
                        'natural_language_query': natural_language_query,
                        'json_query': json_query_str,
                        'results': []
                    })
                collection = db[target_collection]
                logger.debug(f"Executing find query on collection: {target_collection} with query: {json_query}")
                results_cursor = collection.find(json_query).limit(DISPLAY_LIMIT)

            results = list(results_cursor)

            # Convert ObjectId to string and truncate string fields
            truncated_results = [truncate_string_fields(convert_objectids(result), MAX_FIELD_LENGTH) for result in results]

            # Flatten the results for display
            flattened_results = [flatten_dict(result) for result in truncated_results]

            logger.debug(f"Query returned {len(flattened_results)} results (showing first {DISPLAY_LIMIT}): {flattened_results[:3]}")

            if not flattened_results:
                error = 'No results found.'
            else:
                error = ''

            # Store the full query in session for CSV download
            request.session['current_query'] = {
                'json_query': json_query_str,
                'target_collection': target_collection
            }

        except Exception as e:
            error = f"An error occurred while executing the query: {str(e)}"
            logger.error(f"Query Execution Error: {e}")
            return render(request, 'new_query.html', {
                'error': error,
                'natural_language_query': natural_language_query,
                'json_query': json_query_str,
                'results': []
            })

        context = {
            'results': flattened_results,
            'json_query': json_query_str,
            'natural_language_query': natural_language_query,
            'error': error,
            'target_collection': target_collection
        }
        return render(request, 'query_result.html', context)

    return render(request, 'new_query.html', {
        'error': '',
        'natural_language_query': '',
        'json_query': '',
        'results': []
    })

def download_csv(request):
    """
    Generates a CSV file containing all results of the current query and sends it as a downloadable file.
    """
    # Retrieve the current query from the session
    current_query = request.session.get('current_query', None)

    if not current_query:
        error_message = "No query found in session. Please execute a query first."
        logger.error(error_message)
        return HttpResponse(error_message, content_type='text/plain')

    json_query_str = current_query.get('json_query', '').strip()
    target_collection = current_query.get('target_collection', '').strip()

    logger.debug(f"Downloading CSV for Query: {json_query_str} on Collection: {target_collection}")

    try:
        json_query = json.loads(json_query_str)
        logger.debug(f"Parsed JSON Query for CSV: {json_query}")
    except json.JSONDecodeError as e:
        error = f'Invalid JSON query format: {str(e)}'
        logger.error(error)
        return HttpResponse(error, content_type='text/plain')

    is_aggregate = isinstance(json_query, list)
    db = get_db()

    try:
        if is_aggregate:
            if not target_collection:
                error = 'Target collection must be specified for aggregation pipelines.'
                logger.error(error)
                return HttpResponse(error, content_type='text/plain')
            collection = db[target_collection]
            logger.debug(f"Executing aggregation pipeline on collection: {target_collection} for CSV download")
            results_cursor = collection.aggregate(json_query)
        else:
            if not target_collection:
                error = 'Target collection must be specified for single-field queries.'
                logger.error(error)
                return HttpResponse(error, content_type='text/plain')
            collection = db[target_collection]
            logger.debug(f"Executing find query on collection: {target_collection} with query: {json_query}")
            results_cursor = collection.find(json_query)

        results = list(results_cursor)

        if not results:
            error_message = "No results found for the current query."
            logger.warning(error_message)
            return HttpResponse(error_message, content_type='text/plain')

        # Convert all ObjectId instances to strings and flatten the results
        flattened_results = [flatten_dict(truncate_string_fields(convert_objectids(result))) for result in results]

        # Create the HttpResponse object with CSV headers
        response = HttpResponse(content_type='text/csv')
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        response['Content-Disposition'] = f'attachment; filename="query_results_{timestamp}.csv"'

        writer = csv.writer(response)

        # Write CSV Header based on keys of the first result
        headers = flattened_results[0].keys()
        writer.writerow(headers)

        # Write data rows
        for result in flattened_results:
            row = []
            for key in headers:
                value = result.get(key, '')
                value = str(value)
                row.append(value)
            writer.writerow(row)

        logger.info(f"CSV download successful with {len(results)} records.")

        # Do not clear the current_query from session to allow repeated downloads
        return response

    except Exception as e:
        error = f"An error occurred while generating the CSV: {str(e)}"
        logger.error(f"CSV Generation Error: {e}")
        return HttpResponse(error, content_type='text/plain')


def examples(request):
    sample_queries = [
        {
            'natural_language': 'Find plasmids with mobilizable mobility.',
            'json_query': '{"mobility": "mobilizable"}',
            'collection': 'plasmids',
            'difficulty': 20  
        },
        {
            'natural_language': 'Find plasmids with size greater than 10000 bp.',
            'json_query': '{"size": {"$gt": 10000}}',
            'collection': 'plasmids',
            'difficulty': 20  
        },
        {
            'natural_language': 'Find the 10 most common resistance genes in Salmonella.',
            'json_query': '[{"$lookup":{"from":"plasmids","localField":"plasmid_id","foreignField":"_id","as":"plasmid_info"}},{"$unwind":"$plasmid_info"},{"$lookup":{"from":"hosts","localField":"plasmid_info.host_id","foreignField":"_id","as":"host_info"}},{"$unwind":"$host_info"},{"$match":{"host_info.genus":"Salmonella","gene_name":{"$ne":null},"antibiotic_resistance":true}},{"$group":{"_id":"$resistance_info.gene_name","count":{"$sum":1}}},{"$sort":{"count":-1}},{"$limit":10}]',
            'collection': 'genes',
            'difficulty': 60
        },
        {
            'natural_language': 'How many times do plasmids that include "IncQ" in the replicon type, but NOT a comma, appear in Escherichia?.',
            'json_query': '[{ "$match": { "$and": [ { "replicon_type": { "$regex": "IncQ", "$options": "i" } }, { "replicon_type": { "$not": { "$regex": "," } } } ] } },'
                          '{ "$lookup": { "from": "hosts", "localField": "host_id", "foreignField": "_id", "as": "host_info" } },'
                          '{ "$unwind": "$host_info" },'
                          '{ "$match": { "host_info.genus": "Escherichia" } },'
                          '{ "$count": "num_plasmids_with_IncQ_and_no_comma_in_Escherichia" }]',
            'collection': 'plasmids',
            'difficulty': 90  
        }
    ]

    # Truncate natural_language_query for display purposes only
    truncated_sample_queries = []
    for query in sample_queries:
        truncated_query = {
            'natural_language': (query['natural_language'][:MAX_QUERY_LENGTH] 
                                 if 'natural_language' in query else ''),
            'json_query': query['json_query'],  # Do not truncate JSON queries
            'collection': query['collection'],
            'difficulty': query['difficulty']
        }
        truncated_sample_queries.append(truncated_query)

    logger.debug(f"Sample queries: {truncated_sample_queries}")  # Debug: Check if the list is correctly defined
    return render(request, 'examples.html', {'sample_queries': truncated_sample_queries})
