import os
import sys
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    faq_path = os.path.join(base_dir, "data/questions/Dataset_QandA.xlsx")
    output_path = os.path.join(base_dir, "evaluation/benchmark_queries.json")
    
    print("Generating benchmark queries from FAQ dataset...")
    
    if not os.path.exists(faq_path):
        print(f"[ERROR] FAQ spreadsheet not found at: {faq_path}")
        return
        
    df = pd.read_excel(faq_path)
    
    # We will curate 30 representative benchmark test cases
    # spanning different categories and difficulty levels
    benchmark = []
    
    # Let's pick some key rows to represent direct queries
    # Row index mappings or keyword search to cover all categories
    categories = df['Question type'].unique() if 'Question type' in df.columns else ['General']
    
    # Map questions by category
    cat_groups = df.groupby('Question type') if 'Question type' in df.columns else { 'General': df }
    
    test_idx = 1
    
    # 1. Add Direct Queries (Exact FAQ matches)
    print("Adding direct queries...")
    for cat, group in cat_groups:
        # Pick 2 questions per category
        sample_rows = group.head(2)
        for _, row in sample_rows.iterrows():
            benchmark.append({
                "id": f"Q_{test_idx:03d}",
                "query": str(row['Question']).strip(),
                "ground_truth_answer": str(row['Answer']).strip(),
                "ground_truth_id": int(row['No.']) if 'No.' in df.columns else test_idx,
                "category": str(cat).strip(),
                "type": "direct"
            })
            test_idx += 1
            
    # 2. Add Rephrased Queries (Natural language variations with typos or synonyms)
    print("Adding rephrased queries...")
    rephrasings = [
        ("how does onboarding work", "How does onboarding work?", 1),
        ("smartlogi budgt how much", "What is the total budget for the project?", 1),
        ("where is release doc", "What is the release checklist for the project?", 56), # hypothetical index matching typical release checklists
        ("who is manager", "Who is the project manager?", 2),
        ("smartlogi technology stack", "What technologies are used in the project?", 3),
        ("qa test environment link", "How to access the QA environment?", 24)
    ]
    
    for query, matched_question, faq_no in rephrasings:
        # Find the correct answer from the dataframe if possible
        ans_rows = df[df['No.'] == faq_no] if 'No.' in df.columns else df[df['Question'].str.contains(query[:10], case=False, na=False)]
        if not ans_rows.empty:
            ans = str(ans_rows.iloc[0]['Answer']).strip()
            cat = str(ans_rows.iloc[0]['Question type']).strip() if 'Question type' in df.columns else "General"
            benchmark.append({
                "id": f"Q_{test_idx:03d}",
                "query": query,
                "ground_truth_answer": ans,
                "ground_truth_id": int(faq_no),
                "category": cat,
                "type": "rephrased"
            })
            test_idx += 1

    # 3. Add Ambiguous Queries (Trigger Intent Clarification)
    print("Adding ambiguous queries...")
    ambiguous_cases = [
        {
            "query": "onboarding",
            "clarified_options": [
                "What are the key steps outlined in our software project's onboarding documentation for new team members?",
                "Can you identify any specific user training modules mentioned within our current system requirements document that pertain to product onboarding processes?",
                "Are there established testing protocols detailed in our test plan documents related to the effectiveness and efficiency of end-user software onboarding procedures?"
            ],
            "type": "ambiguous"
        },
        {
            "query": "search order feature",
            "clarified_options": [
                "[Understanding the functionality of search within an e-commerce platform]",
                "[Implementing a keyword filtering system for orders in inventory management software]",
                "[Optimizing order retrieval performance on customer relationship management (CRM) systems]"
            ],
            "type": "ambiguous"
        },
        {
            "query": "testing guidelines",
            "clarified_options": [
                "QA environment setup and credentials",
                "Unit testing framework and commands",
                "User acceptance testing (UAT) templates"
            ],
            "type": "ambiguous"
        }
    ]
    
    for case in ambiguous_cases:
        benchmark.append({
            "id": f"Q_{test_idx:03d}",
            "query": case["query"],
            "clarified_options": case["clarified_options"],
            "type": "ambiguous",
            "category": "Ambiguity"
        })
        test_idx += 1
        
    # Write to JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Successfully generated {len(benchmark)} benchmark queries saved to {output_path}")

if __name__ == "__main__":
    main()
