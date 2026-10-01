"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline:
  classify intent -> (verified template | generate SQL) -> validate (5 checks)
  -> retry once (generated track only) -> escalate if still failing
  -> execute (read-only) -> generate narrative -> audit log
"""

import json
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Configuration (paths can be overridden with environment variables)
# ----------------------------------------------------------------------------
DEFAULT_DB_PATH = os.getenv("CREDIT_RISK_DB_PATH", "credit_risk_portfolio.db")
DEFAULT_TEST_CSV_PATH = os.getenv("TEST_QUERIES_CSV_PATH", "test_queries.csv")
CONFIG_JSON_PATH = os.getenv("CONFIG_JSON_PATH", "config.json")  # optional fallback
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit_log.jsonl")

# ----------------------------------------------------------------------------
# Database schema provided to the LLM in prompts (unchanged from notebook)
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ----------------------------------------------------------------------------
# Verified Query Templates (unchanged from notebook)
# ----------------------------------------------------------------------------
sql_1 = """SELECT
  sm.sector_name,
  SUM(lm.total_outstanding) / 1000000 AS total_outstanding_mil,
  SUM(CASE WHEN lm.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN lm.total_outstanding ELSE 0 END) / 1000000 AS npa_outstanding_mil
FROM loan_master AS lm
JOIN sector_master AS sm
  ON lm.sector_code = sm.sector_code
GROUP BY
  sm.sector_name
ORDER BY
  total_outstanding_mil DESC;"""

sql_2 = """SELECT
  loan_category,
  SUM(total_outstanding) / 1000000 AS total_outstanding_mil,
  COUNT(loan_account_number) AS loan_count
FROM
  loan_master
GROUP BY
  loan_category
ORDER BY
  total_outstanding_mil DESC;"""

sql_3 = """SELECT
  ifrs9_stage,
  COUNT(loan_account_number) AS loan_count,
  SUM(ead_amount) / 1000000 AS total_ead_mil,
  SUM(ecl_amount) / 1000000 AS total_ecl_mil
FROM
  provisioning
WHERE
  reporting_date = '2025-09-30'
GROUP BY
  ifrs9_stage
ORDER BY
  ifrs9_stage;"""

sql_4 = """SELECT
  sm.sector_name,
  AVG(p.provision_coverage_ratio) AS average_provision_coverage_ratio
FROM
  provisioning AS p
JOIN
  loan_master AS lm
  ON p.loan_account_number = lm.loan_account_number
JOIN
  sector_master AS sm
  ON lm.sector_code = sm.sector_code
WHERE
  p.reporting_date = '2025-09-30'
GROUP BY
  sm.sector_name
ORDER BY
  average_provision_coverage_ratio DESC;"""

sql_5 = """SELECT
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000 AS total_outstanding_mil,
  lm.asset_classification
FROM
  loan_master AS lm
JOIN
  sector_master AS sm
  ON lm.sector_code = sm.sector_code
ORDER BY
  total_outstanding_mil DESC
LIMIT 10;"""

sql_6 = """SELECT
  group_name,
  COUNT(loan_account_number) AS loan_count,
  SUM(total_outstanding) / 1000000 AS total_outstanding_mil
FROM
  loan_master
WHERE
  group_name IS NOT NULL
GROUP BY
  group_name
ORDER BY
  total_outstanding_mil DESC
LIMIT 5;"""

sql_7 = """SELECT
  lm.loan_account_number,
  lm.borrower_name,
  sm.sector_name,
  lm.total_outstanding / 1000000 AS total_outstanding_mil,
  lm.days_past_due,
  lm.asset_classification
FROM
  loan_master AS lm
JOIN
  sector_master AS sm
  ON lm.sector_code = sm.sector_code
WHERE
  lm.days_past_due > 0
ORDER BY
  lm.days_past_due DESC;"""

sql_8 = """SELECT
  CASE
    WHEN days_past_due = 0 THEN '0 (Current)'
    WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
    WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
    WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
    ELSE '90+'
  END AS dpd_bucket,
  COUNT(loan_account_number) AS loan_count,
  SUM(total_outstanding) / 1000000 AS total_outstanding_mil
FROM
  loan_master
GROUP BY
  dpd_bucket
ORDER BY
  CASE
    WHEN dpd_bucket = '0 (Current)' THEN 1
    WHEN dpd_bucket = '1-30' THEN 2
    WHEN dpd_bucket = '31-60' THEN 3
    WHEN dpd_bucket = '61-90' THEN 4
    ELSE 5
  END;"""

sql_9 = """SELECT
  borrower_id,
  previous_rating,
  internal_rating,
  pd_estimate
FROM
  borrower_rating
WHERE
  rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY
  pd_estimate DESC;"""

sql_10 = """SELECT
  reporting_date,
  SUM(ecl_amount) / 1000000 AS total_ecl_mil
FROM
  provisioning
GROUP BY
  reporting_date
ORDER BY
  reporting_date;"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}

# ----------------------------------------------------------------------------
# Credentials, LLMs and DB connection
# ----------------------------------------------------------------------------
def load_credentials():
    """
    Resolve OPENAI_API_KEY / OPENAI_API_BASE.
    Priority: st.secrets -> environment variables -> config.json (as in notebook).
    """
    api_key, api_base = None, None

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        api_base = st.secrets.get("OPENAI_API_BASE")
    except Exception:
        pass  # no secrets file present

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    api_base = api_base or os.environ.get("OPENAI_API_BASE")

    if (not api_key) and os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r") as f:
                cfg = json.load(f)
            api_key = cfg.get("OPENAI_API_KEY")
            api_base = api_base or cfg.get("OPENAI_API_BASE")
        except Exception:
            pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    return api_key, api_base


@st.cache_resource(show_spinner=False)
def get_llms(api_key, api_base):
    kwargs = dict(temperature=0, openai_api_key=api_key)
    if api_base:
        kwargs["openai_api_base"] = api_base
    llm_ = ChatOpenAI(model='gpt-4o-mini', **kwargs)
    evaluator_llm_ = ChatOpenAI(model='gpt-4o', **kwargs)
    return llm_, evaluator_llm_


@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    # Read-only connection (URI mode) - prevents any write operations
    return sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, check_same_thread=False)


# ----------------------------------------------------------------------------
# Tool 1: Intent classification
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Returns:
    - dict: 'route' (verified or generated), 'query_id' (template ID or None),
            'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
Given the user's question and the following verified query library descriptions:

Verified Query Library Descriptions:
{library_descriptions}

User Question: {user_question}

Your task is to classify the user's intent. If the question can be answered by an existing verified query, return its 'query_id'. If no exact match is found, return null for 'query_id'. Provide a concise reason for your classification.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "a concise reason for your classification"
}}
Do not include any other text or explanation outside the JSON.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        classification_json = json.loads(json_match.group())
        query_id = classification_json.get('query_id')
        match_reason = classification_json.get('match_reason', 'No specific reason provided.')

        if query_id and query_id in query_library:
            return {"route": "verified", "query_id": query_id, "match_reason": match_reason}
        else:
            return {"route": "generated", "query_id": None, "match_reason": match_reason}
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ----------------------------------------------------------------------------
# Tool 2: Query generation
# ----------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.
    '''

    generation_prompt = f"""
You are a SQL expert. Your task is to write a single, read-only SQLite SQL query that answers the user's question.
Use the provided database schema to construct the query.

Database Schema:
{schema_context}

User Question: {user_question}

Output ONLY the SQL query, nothing else. Do NOT include any explanations, introductory phrases, conversational text, or markdown code fences (```sql).
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Robustly strip markdown fences if present, despite prompt instruction
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ----------------------------------------------------------------------------
# Tool 3: Query validation (five checks)
# ----------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Returns:
    - dict: 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (float, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    # (Same logic as the notebook: unknown identifiers are computed but, as in the
    #  original implementation, are not used to fail the query.)
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""
As an expert SQL query validator, your task is to determine if the provided SQL query accurately answers the user's question based on the given context.

Context:
{track_context}

User Question: {user_question}

Candidate SQL:
{candidate_sql}

IMPORTANT: Return ONLY a JSON dictionary with these exact keys. Do not include any other text or explanation.
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence explaining your verdict and confidence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # Use sqlparse to get a clean, single statement, then strip trailing semicolon
            expected_clean_sql = sqlparse.parse(expected_sql)[0].value.strip().rstrip(';')
            candidate_clean_sql = sqlparse.parse(candidate_sql)[0].value.strip().rstrip(';')

            expected_cols = [d[0] for d in cur.execute(f"{expected_clean_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_clean_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ----------------------------------------------------------------------------
# Tool 4: Retry generation
# ----------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.
    '''

    retry_prompt = f"""
A previously generated SQL query failed validation. Your task is to revise the SQL query to fix the error and answer the user's question.
The revised query must be a single, read-only SQLite SQL query.

Database Schema:
{schema_context}

User Question: {user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Output ONLY the revised SQL query, nothing else. Do NOT include any explanations, introductory phrases, conversational text, or markdown code fences (```sql).
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ----------------------------------------------------------------------------
# Tool 5: Query execution
# ----------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Returns:
    - dict: 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ----------------------------------------------------------------------------
# Tool 6: Response generation
# ----------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.
    '''

    response_prompt = f"""
Based on the user's question and the provided data, generate a concise, business-focused natural language response.
Highlight the most relevant insights and exact figures from the data.

User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

Natural Language Response:
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ----------------------------------------------------------------------------
# Pipeline orchestration
# ----------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, trace=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - verbose (bool): If True, pipeline stages are emitted.
    - trace (list, optional): if provided, stage messages are appended to it
      (used by the Streamlit UI); otherwise they are printed to the console.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def emit(msg):
        if not verbose:
            return
        if trace is not None:
            trace.append(msg)
        else:
            print(msg)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")
    log['warnings'] = exec_result['warnings']

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ----------------------------------------------------------------------------
# Audit trail helpers
# ----------------------------------------------------------------------------
def append_audit(log):
    """Append the full pipeline log (without the DataFrame) to a JSONL audit file."""
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **log}
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        st.warning(f"Could not write audit log to '{AUDIT_LOG_PATH}': {e}")
    return entry


def history_row(entry):
    conf = entry.get("confidence")
    return {
        "Timestamp": entry["timestamp"],
        "Question": entry["user_question"],
        "Route": entry["route"],
        "Query ID": entry["query_id"],
        "Confidence": conf,
        "Rows": entry["row_count"],
        "Retry used": entry["retry_used"],
        "Escalated": entry["escalated"],
    }


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def render_result(res):
    """Display narrative, confidence, SQL, data, and validation details."""
    log = res["log"]

    if log["escalated"]:
        st.error("⚠️ Escalated to a human analyst")
        st.write(log["narrative"])
    else:
        st.subheader("Answer")
        st.write(log["narrative"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", str(log["route"]).capitalize())
    c2.metric("Query ID", log["query_id"] if log["query_id"] else "—")
    conf = log["confidence"]
    if isinstance(conf, (int, float)):
        c3.metric("Confidence", f"{conf:.2f}")
    else:
        c3.metric("Confidence", str(conf) if conf is not None else "—")
    c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "—")

    if isinstance(conf, (int, float)):
        st.progress(min(max(float(conf), 0.0), 1.0))

    st.caption(f"Routing reason: {log['match_reason']}")
    if log["retry_used"]:
        st.info("A retry was used: the first generated SQL failed validation and was regenerated once.")

    st.subheader("SQL used")
    sql_to_show = log["executed_sql"] or log["candidate_sql"]
    st.code(sql_to_show, language="sql")

    st.subheader("Raw data returned")
    if res["dataframe"] is not None:
        st.dataframe(res["dataframe"])
        st.download_button(
            "Download result as CSV",
            data=res["dataframe"].to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
        )
    else:
        st.write("No data returned (query was escalated).")

    for w in log.get("warnings", []) or []:
        st.warning(w)

    with st.expander("Validation gate details"):
        st.json(log["gate_result"])

    if res.get("trace"):
        with st.expander("Pipeline trace"):
            st.code("\n".join(res["trace"]))


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Northbridge Bank · Commercial lending portfolio · Read-only natural-language analytics "
    "with verified SQL templates, validation gates and a full audit trail."
)

# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    db_path = st.text_input("Database path", value=DEFAULT_DB_PATH)
    test_csv_path = st.text_input("Test queries CSV path", value=DEFAULT_TEST_CSV_PATH)
    st.markdown("---")
    st.markdown(
        "**Models**\n\n- Pipeline LLM: `gpt-4o-mini`\n- Validator LLM: `gpt-4o`\n\n"
        "**Mode:** read-only (SQLite `mode=ro`)"
    )

# ---- Credentials / LLMs ----------------------------------------------------
api_key, api_base = load_credentials()
if not api_key:
    st.error(
        "OPENAI_API_KEY not found. Add it to `.streamlit/secrets.toml` (or Streamlit Cloud "
        "secrets), set it as an environment variable, or provide a `config.json`."
    )
    st.stop()

llm, evaluator_llm = get_llms(api_key, api_base)

# ---- Database --------------------------------------------------------------
if not os.path.exists(db_path):
    st.error(f"Database file not found at '{db_path}'. Place `credit_risk_portfolio.db` next to app.py or update the path in the sidebar.")
    st.stop()
if os.path.getsize(db_path) == 0:
    st.error(f"Database file at '{db_path}' is empty.")
    st.stop()

try:
    conn = get_connection(db_path)
except Exception as e:
    st.error(f"Could not open database: {e}")
    st.stop()

with st.sidebar:
    try:
        tbls = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        st.success(f"Connected (read-only). Tables: {', '.join(tbls)}")
    except Exception as e:
        st.error(f"Connection check failed: {e}")

# ---- Ground truth (optional) ----------------------------------------------
ground_truth = None
if os.path.exists(test_csv_path):
    try:
        ground_truth = pd.read_csv(test_csv_path)
    except Exception as e:
        st.sidebar.warning(f"Could not read test CSV: {e}")
else:
    st.sidebar.info("Test CSV not found - evaluation tab and sample questions are disabled.")

# ---- Session state ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

tab_ask, tab_eval, tab_lib, tab_audit = st.tabs(
    ["💬 Ask a Question", "🧪 Evaluation", "📚 Verified Query Library", "🧾 Audit Log"]
)

# ============================== Tab 1: Ask ===================================
with tab_ask:
    PLACEHOLDER = "— type my own question —"
    sample_options = [PLACEHOLDER]
    if ground_truth is not None and "User Query" in ground_truth.columns:
        sample_options += ground_truth["User Query"].dropna().tolist()

    sample = st.selectbox("Sample questions", sample_options)
    question = st.text_input(
        "Ask a question about the commercial lending portfolio",
        value="" if sample == PLACEHOLDER else sample,
        placeholder="e.g. Show me the total outstanding exposure for all sectors.",
    )

    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            trace = []
            with st.spinner("Running pipeline..."):
                try:
                    res = run_pipeline(
                        question.strip(), conn, verified_query_library, database_schema,
                        verbose=True, trace=trace,
                    )
                    res["trace"] = trace
                    entry = append_audit(res["log"])
                    st.session_state["history"].append(entry)
                    st.session_state["last_result"] = res
                except Exception as e:
                    st.session_state["last_result"] = None
                    st.error(f"Pipeline error: {e}")

    if st.session_state["last_result"] is not None:
        render_result(st.session_state["last_result"])

# ============================== Tab 2: Evaluation ============================
with tab_eval:
    st.subheader("Evaluation against ground truth")
    st.write(
        "Runs every test case in the test-queries CSV through the pipeline and reports "
        "Selected Path Accuracy, Selected Query Accuracy and Average Confidence."
    )

    if ground_truth is None:
        st.info("Provide `test_queries.csv` (columns: Test Case, User Query, Expected Route, Expected Query ID, Expected Answer).")
    else:
        st.dataframe(ground_truth)

        if st.button("Run all test cases"):
            test_results = []
            progress = st.progress(0.0)
            status = st.empty()
            n = len(ground_truth)

            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                status.write(f"Running {gt['Test Case']} ({i + 1}/{n})...")
                try:
                    tr = run_pipeline(gt["User Query"], conn, verified_query_library,
                                      database_schema, verbose=False)
                    append_audit(tr["log"])
                except Exception as e:
                    tr = {"route": f"error: {e}", "query_id": None, "confidence": None,
                          "row_count": None, "dataframe": None, "narrative": str(e),
                          "executed_sql": None}
                test_results.append(tr)
                progress.progress((i + 1) / n)
            status.empty()

            evaluation_rows = []
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = test_results[i]
                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            # Convert 'ESCALATED' strings in 'Confidence' column to NaN for numeric calculation
            evaluation_df['Confidence'] = evaluation_df['Confidence'].apply(pd.to_numeric, errors='coerce')

            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

            average_confidence = evaluation_df['Confidence'].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
            m2.metric("Selected Query Accuracy", f"{query_accuracy:.1f}%")
            m3.metric("Average Confidence Score", f"{average_confidence:.2f}")

            st.dataframe(evaluation_df)

            st.subheader("Per-test-case details")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = test_results[i]
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    st.write(f"**Confidence:** {tr['confidence']}")
                    st.write("**Narrative:**")
                    st.write(tr["narrative"])
                    st.write("**Executed SQL:**")
                    st.code(tr["executed_sql"] or "(none)", language="sql")
                    st.write("**Result data:**")
                    if tr["dataframe"] is not None:
                        st.dataframe(tr["dataframe"])
                    if "Expected Answer" in ground_truth.columns:
                        st.write("**Expected answer (ground truth):**")
                        st.write(gt["Expected Answer"])

# ============================== Tab 3: Library ===============================
with tab_lib:
    st.subheader("Verified Query Template Library")
    st.write(f"{len(verified_query_library)} pre-approved, version-controlled SQL templates.")
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"], language="sql")

    with st.expander("Database schema provided to the LLM"):
        st.code(database_schema)

# ============================== Tab 4: Audit =================================
with tab_audit:
    st.subheader("Audit trail")
    st.caption(f"Every pipeline run is appended to `{AUDIT_LOG_PATH}` (JSON Lines).")

    if st.session_state["history"]:
        st.write("**This session**")
        st.dataframe(pd.DataFrame([history_row(e) for e in st.session_state["history"]]))
        st.download_button(
            "Download session audit log (JSON)",
            data=json.dumps(st.session_state["history"], indent=2, default=str).encode("utf-8"),
            file_name="session_audit_log.json",
            mime="application/json",
        )
    else:
        st.info("No queries have been run in this session yet.")

    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "rb") as f:
            st.download_button(
                "Download full audit log file (JSONL)",
                data=f.read(),
                file_name="audit_log.jsonl",
                mime="application/jsonl",
            )
