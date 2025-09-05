from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from groq import Groq
import re
import requests
import os

# ----------------- Direct API Key Declaration -----------------
api_key = "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"  # Replace with your actual Groq API key

if not api_key or api_key.strip() == "":
    raise ValueError("❌ No API key found! Please add your Groq API key to the api_key variable.")

# ----------------- Server Check -----------------
def check_groq_server(api_key, model="llama-3.1-8b-instant"):
    client = Groq(api_key=api_key)
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model=model,
            max_tokens=1,
            temperature=0.1
        )
        if response.choices[0].message.content:
            return True
    except Exception:
        return False
    return False

# ----------------- Initialize LLM -----------------
server_status = check_groq_server(api_key)
llm_engine = None
if server_status:
    try:
        llm_engine = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_retries=2,
            timeout=30
        )
    except Exception as e:
        print(f"⚠ Failed to init LLM: {e}")
        server_status = False

# ----------------- PROMPTS -----------------
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a healthcare professional generating a personalized 7-day care plan. "
    "Recommendations must be based ONLY on the provided patient report text (not on generic assumptions). "
    "Create varied and personalized daily recommendations while maintaining medical appropriateness. "
    "Ensure all recommendations are practical and directly tailored to the conditions explicitly identified in the report."
)

def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def _extract_plan_only(text: str) -> str:
    header_idx = text.find("Enhanced 7-Day Care Plan")
    if header_idx == -1:
        header_idx = text.find("7-Day Care Plan")
    if header_idx == -1:
        header_idx = text.find("Day 1")
    if header_idx == -1:
        return text.strip()
    return text[header_idx:].strip()

def _format_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r'(Day \d+)', r'\n\n\1\n', text)
    text = re.sub(r'(🏃[^🧘🥗💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(🧘[^🥗💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(🥗[^💧❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(💧[^❌✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(❌[^✅⚠]+)', r'\1\n', text)
    text = re.sub(r'(✅[^⚠]+)', r'\1\n', text)
    text = re.sub(r'(⚠.*?)(?=\nDay|\Z)', r'\1\n', text, flags=re.DOTALL)  # ✅ fixed line
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _validate_plan(text: str) -> None:
    required = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    for d in required:
        if d not in text:
            raise ValueError("Plan missing days")

def extract_conditions_from_report(report_text: str) -> dict:
    """Extract all medical conditions from the report"""
    conditions = {}
    
    # Extract diseases with risk scores
    disease_pattern = r"- Disease: (.?)\n.?Risk Score: (.?)\n.?Reasoning: (.*?)(?=\n\n|\n- Disease:|$)"
    matches = re.findall(disease_pattern, report_text, re.DOTALL)
    
    for disease, risk_score, reasoning in matches:
        clean_disease = disease.strip()
        conditions[clean_disease] = {
            'risk_score': risk_score.strip(),
            'reasoning': reasoning.strip()[:200] + "..." if len(reasoning) > 200 else reasoning.strip()
        }
    
    # Extract diabetes type
    diabetes_type_match = re.search(r"Predicted Diabetes Type: (.*?)\n", report_text)
    diabetes_confidence_match = re.search(r"Confidence Score: (.*?)\n", report_text)
    
    if diabetes_type_match:
        diabetes_type = diabetes_type_match.group(1).strip()
        conditions["Diabetes"] = {
            'type': diabetes_type,
            'confidence': diabetes_confidence_match.group(1).strip() if diabetes_confidence_match else "High",
            'reasoning': "Predicted from clinical indicators in report"
        }
    
    return conditions

def _build_prompt_from_report(report_text: str) -> str:
    """Build prompt based on the exact report, no generic additions"""
    conditions = extract_conditions_from_report(report_text)
    
    # Summarize conditions
    conditions_summary = []
    for condition, details in conditions.items():
        summary = f"- {condition}: "
        if 'risk_score' in details:
            summary += f"Risk: {details['risk_score']}, "
        if 'type' in details:
            summary += f"Type: {details['type']}, "
        summary += f"Reason: {details.get('reasoning', 'Not specified')}"
        conditions_summary.append(summary)
    
    conditions_text = "\n".join(conditions_summary) if conditions_summary else "No specific conditions identified"
    
    return f"""As a healthcare professional, generate a personalized 7-day care plan for this patient.

PATIENT REPORT:
{report_text}

IDENTIFIED MEDICAL CONDITIONS:
{conditions_text}

GUIDELINES:
- Only use the information provided in the patient report (no assumptions, no generic conditions).
- Address ALL identified conditions with targeted care recommendations.
- Vary activities, meals, and suggestions each day.
- Keep this structure for each day:

Day [X]  
🏃 Physical Activity: [Specific activity] → [How it helps conditions]  
🧘 Mental Wellness: [Specific activity] → [How it helps conditions]  
🥗 Meals: [Specific meals] → [How it helps conditions]  
💧 Hydration: [Plan] → [How it helps conditions]  
❌ Avoid: [Items/behaviors] → [Risks if taken]  
✅ Today's risk reduction: [How today’s plan reduces risks]  
⚠ Consequences if skipped: [Realistic risks like worsening health, poor sugar control,or hypertension etc.]"""

def generate_care_plan_from_report(report_text: str) -> str:
    """Generate 7‑day plan via LLM. If LLM is unreachable, raise a clear error."""
    if llm_engine is None:
        raise RuntimeError("Care plan server busy, try again")
    detailed_prompt = _build_prompt_from_report(report_text)
    try:
        prompt_template = ChatPromptTemplate.from_messages([
            system_prompt,
            HumanMessagePromptTemplate.from_template(detailed_prompt)
        ])
        chain = prompt_template | llm_engine | StrOutputParser()
        response = chain.invoke({})
        text = _strip_think_tags(response)
        text = _extract_plan_only(text)
        text = _format_output(text)
        _validate_plan(text)
        return text
    except Exception as e:
        raise RuntimeError("Care plan server busy, try again") from e

# ----------------- Variable Input -----------------
report_input = """--- Enter Patient Data ---
Gender (Male/Female):  male
Age (years):  75
BMI:  29
Smoking Status (Never/Former/Current):  former
History of stroke? (0/1) (0/1):  1
Systolic BP:  185
Diastolic BP:  110
Heart Rate:  85
Respiratory Rate:  18
Fasting Blood Sugar:  110
HbA1c:  5.8
Serum Creatinine:  1.1
eGFR:  75
BUN:  20
Total Cholesterol:  220
LDL Cholesterol:  155
HDL Cholesterol:  45
Triglycerides:  160
Hemoglobin:  14.2
Urine Albumin ACR:  25
Glucose in Urine? (0/1) (0/1):  0
FEV1/FVC Ratio:  0.81

--- Final Diagnostic Report ---
The model predicts the patient may have the following conditions:

- Disease: Heart Disease
  Risk Score: 94.6/100 (High)
  Reasoning: high LDL ('bad') cholesterol (155.0 mg/dL), high triglycerides (160.0 mg/dL), presence of high blood pressure, history of smoking

- Disease: Hypertension
  Risk Score: 95.0/100 (High)
  Reasoning: high blood pressure (Stage 2) at 185/110 mmHg, advanced age

- Disease: Stroke
  Risk Score: 100.0/100 (High)
  Reasoning: a prior history of stroke, critically high blood pressure (185/110 mmHg), uncontrolled hypertension
  Model Accuracy: 99.00%



--- Please Enter Patient Clinical Data ---

Is the patient pregnant? (0 for No, 1 for Yes): 0

Age at Diagnosis: 60

BMI at Diagnosis (e.g., 25.4): 38.1

HbA1c level (e.g., 6.5): 9.1

C-Peptide Level (e.g., 0.8): 1.8

Family History:

1. Strong_Multi_Generational

2. nan

3. Parent/Sibling_T2D

4. Parent/Sibling_T1D

Enter your choice (number or text): 3

Autoantibodies Status:

1. Negative

2. GAD65_Positive

3. Multiple_Positive

Enter your choice (number or text): 1

Genetic Test Result:

1. Known_MODY_Mutation

2. Negative

Enter your choice (number or text): 2

---------------------------------------------

Diabetes Prediction Report

---------------------------------------------

Predicted Diabetes Type: T2D

Confidence Score: 100.00%



Explanation:

Prediction reasoning based on clinical indicators:

- Absence of autoantibodies and high BMI (38.1) indicate insulin resistance type diabetes.

- Adult age at diagnosis (60).





Predicted Diabetes Risk Level: High

Cluster Stats: Mean HbA1c: 9.57, BMI: 33.9, Age: 55.3



Risk Explanation:

Risk level is derived by comparing your clinical factors to clusters of patients.

Higher HbA1c and BMI in the high-risk cluster correspond to higher diabetes risk.

Lower values in the low-risk cluster indicate healthier profiles.
"""
# ----------------- SEPARATE 7 DAYS -----------------
def split_days(plan_text: str):
    """Split care plan into dictionary with Day 1..Day 7"""
    days = {}
    matches = re.split(r"\n(?=Day \d+)", plan_text.strip())  # split at each "Day X"
    for m in matches:
        if m.strip():
            day_label = m.split("\n")[0].strip()
            days[day_label] = m.strip()
    return days

def print_day(day_text: str):
    """Print day content up to and including the full ❌ Avoid sentence, then ask for input"""
    # Find key section indexes
    avoid_idx = day_text.find("❌ Avoid")
    reduction_idx = day_text.find("✅ Today's risk reduction")
    consequences_idx = day_text.find("⚠ Consequences if skipped")

    if avoid_idx != -1:
        # Find the end of the ❌ Avoid line/sentence (stop at newline or end of text)
        end_of_avoid = day_text.find("\n", avoid_idx)
        if end_of_avoid == -1:  # if ❌ Avoid is last line
            end_of_avoid = len(day_text)
        print(day_text[:end_of_avoid].strip())  # ✅ now prints full sentence

    # Ask user for choice
    user_choice = input("Did you follow today's care plan ?").strip().lower()

    if user_choice == "yes" and reduction_idx != -1:
        # Print from ✅ until ⚠
        next_idx = consequences_idx if consequences_idx != -1 else len(day_text)
        print("\n" + day_text[reduction_idx:next_idx].strip())
    elif user_choice == "no" and consequences_idx != -1:
        # Print ⚠ section
        print("\n" + day_text[consequences_idx:].strip())


if __name__ == '__main__':
    # Demo only: generate from the sample report_input
    if server_status and llm_engine:
        try:
            print("Analyzing patient report and generating care plan...")
            care_plan_output = generate_care_plan_from_report(report_input)
            print("✅ Care plan generated successfully!")
            care_plan = care_plan_output
            days_dict = split_days(care_plan)
            day1 = days_dict.get("Day 1", "")
            day2 = days_dict.get("Day 2", "")
            print("\n===== DAY 1 PLAN =====")
            print_day(day1)
            print("\n===== DAY 2 PLAN =====")
            print_day(day2)
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    else:
        print("⚠ Please check your Groq API connection.")