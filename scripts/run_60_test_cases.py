import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set offline flags for Hugging Face to use local cache immediately
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from fastapi.testclient import TestClient
from src.api.api import app, _clarification_sessions, _clarification_sessions_lock

TEST_CASES = [
    # GROUP A — FAQ / RAG
    {
        "Test_ID": "T01",
        "Test_Group": "A",
        "Query": "What is SmartLogi?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T02",
        "Test_Group": "A",
        "Query": "What is the purpose of the approval record?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T03",
        "Test_Group": "A",
        "Query": "What are the main requirements of the CRM system?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T04",
        "Test_Group": "A",
        "Query": "What does the CRM system support?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T05",
        "Test_Group": "A",
        "Query": "What are the requirements for critical CRM services?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T06",
        "Test_Group": "A",
        "Query": "What is the target RTO for critical CRM services?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T07",
        "Test_Group": "A",
        "Query": "What happens when a critical error occurs?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T08",
        "Test_Group": "A",
        "Query": "Explain the CRM reporting requirements.",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T09",
        "Test_Group": "A",
        "Query": "What are the requirements for PDF and Excel report export?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T10",
        "Test_Group": "A",
        "Query": "What does the SRS say about authentication and authorization?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T11",
        "Test_Group": "A",
        "Query": "How does the approval process work?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T12",
        "Test_Group": "A",
        "Query": "What is the purpose of the CRM dashboard?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T13",
        "Test_Group": "A",
        "Query": "Tell me about the CRM system requirements.",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T14",
        "Test_Group": "A",
        "Query": "What are the release requirements described in the documents?",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Hard",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T15",
        "Test_Group": "A",
        "Query": "Give me the relevant CRM specification for system scalability.",
        "Expected_Intent": "GENERAL_RAG",
        "Expected_Target": "",
        "Difficulty": "Hard",
        "Ambiguity": "Clear"
    },
    # GROUP B — SPECIFIC TICKET QUERY
    {
        "Test_ID": "T16",
        "Test_Group": "B",
        "Query": "What is the status of CRM-050?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T17",
        "Test_Group": "B",
        "Query": "Tell me the content of ticket CRM-050.",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T18",
        "Test_Group": "B",
        "Query": "What is the current priority of CRM-050?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T19",
        "Test_Group": "B",
        "Query": "Who is assigned to CRM-050?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T20",
        "Test_Group": "B",
        "Query": "How many story points does CRM-050 have?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T21",
        "Test_Group": "B",
        "Query": "Which sprint does CRM-050 belong to?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T22",
        "Test_Group": "B",
        "Query": "Give me all available information about CRM-050.",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T23",
        "Test_Group": "B",
        "Query": "Show me the details of CRM-044.",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-044",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T24",
        "Test_Group": "B",
        "Query": "What is the status and priority of CRM-060?",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-060",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T25",
        "Test_Group": "B",
        "Query": "Tell me about CRM-999.",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-999",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    },
    # GROUP C — TICKET ANALYSIS
    {
        "Test_ID": "T26",
        "Test_Group": "C",
        "Query": "Explain the project context relevant to CRM-050.",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T27",
        "Test_Group": "C",
        "Query": "Why is CRM-050 important to the project?",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T28",
        "Test_Group": "C",
        "Query": "What dependencies are related to CRM-050?",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T29",
        "Test_Group": "C",
        "Query": "What other tickets are related to CRM-050?",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T30",
        "Test_Group": "C",
        "Query": "Analyze CRM-044 and explain its project impact.",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-044",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T31",
        "Test_Group": "C",
        "Query": "What downstream tasks may depend on CRM-044?",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-044",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T32",
        "Test_Group": "C",
        "Query": "Explain the business impact of CRM-044.",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-044",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T33",
        "Test_Group": "C",
        "Query": "Analyze CRM-060 and explain why it may block other work.",
        "Expected_Intent": "TICKET_ANALYSIS",
        "Expected_Target": "CRM-060",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    # GROUP D — PRIORITY ANALYSIS & RANKING
    {
        "Test_ID": "T34",
        "Test_Group": "D",
        "Query": "Which ticket should we prioritize in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T35",
        "Test_Group": "D",
        "Query": "Which ticket should we prioritize in the Australian Retirement Fund CRM - Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Australian Retirement Fund CRM - Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T36",
        "Test_Group": "D",
        "Query": "Which ticket should we do first in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T37",
        "Test_Group": "D",
        "Query": "Rank the tickets in Sprint CRM Retirement Fund - 2026 - SS02 by priority.",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T38",
        "Test_Group": "D",
        "Query": "What is the most important ticket in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T39",
        "Test_Group": "D",
        "Query": "Which ticket has the highest priority in the current sprint?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "current sprint",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T40",
        "Test_Group": "D",
        "Query": "Recommend the next ticket our team should work on in Sprint CRM Retirement Fund - 2026 - SS02.",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T41",
        "Test_Group": "D",
        "Query": "Analyze the priorities of all tickets in Sprint CRM Retirement Fund - 2026 - SS02.",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Hard",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T42",
        "Test_Group": "D",
        "Query": "Which ticket is the biggest blocker in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Hard",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T43",
        "Test_Group": "D",
        "Query": "Which ticket should be handled first based on dependencies?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "",
        "Difficulty": "Hard",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T44",
        "Test_Group": "D",
        "Query": "What should we work on first?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T45",
        "Test_Group": "D",
        "Query": "Which ticket should we prioritize?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    # GROUP E — INTENT CLARIFICATION
    {
        "Test_ID": "T46",
        "Test_Group": "E",
        "Query": "What should we work on first?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T47",
        "Test_Group": "E",
        "Query": "Which release should I check?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T48",
        "Test_Group": "E",
        "Query": "What does 'release' mean here?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T49",
        "Test_Group": "E",
        "Query": "Tell me about the requirements.",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    {
        "Test_ID": "T50",
        "Test_Group": "E",
        "Query": "What ticket should I work on?",
        "Expected_Intent": "CLARIFICATION_REQUIRED",
        "Expected_Target": "",
        "Difficulty": "Medium",
        "Ambiguity": "Ambiguous"
    },
    # GROUP F — HUMAN APPROVAL / CONTROLLED UPDATE
    {
        "Test_ID": "T51",
        "Test_Group": "F",
        "Query": "Apply the recommended priority to CRM-044.",
        "Expected_Intent": "APPROVAL_REQUEST",
        "Expected_Target": "CRM-044",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T52",
        "Test_Group": "F",
        "Query": "Approve APR-002 by user_pm.",
        "Expected_Intent": "APPROVAL",
        "Expected_Target": "APR-002",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T53",
        "Test_Group": "F",
        "Query": "Reject APR-002.",
        "Expected_Intent": "APPROVAL",
        "Expected_Target": "APR-002",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T54",
        "Test_Group": "F",
        "Query": "Apply APR-002 by user_pm.",
        "Expected_Intent": "CONTROLLED_UPDATE",
        "Expected_Target": "APR-002",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T55",
        "Test_Group": "F",
        "Query": "Change CRM-044 priority without approval.",
        "Expected_Intent": "CONTROLLED_UPDATE",
        "Expected_Target": "CRM-044",
        "Difficulty": "Hard",
        "Ambiguity": "Clear"
    },
    # GROUP G — ROBUSTNESS / OUT-OF-SCOPE / TYPO
    {
        "Test_ID": "T56",
        "Test_Group": "G",
        "Query": "Which ticket should we prioritise in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T57",
        "Test_Group": "G",
        "Query": "Which tcket should we prioritize in Sprint CRM Retirement Fund - 2026 - SS02?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - SS02",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T58",
        "Test_Group": "G",
        "Query": "Tell me the stauts of CRM-050.",
        "Expected_Intent": "TICKET_QUERY",
        "Expected_Target": "CRM-050",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T59",
        "Test_Group": "G",
        "Query": "Which ticket should we prioritize in Sprint CRM Retirement Fund - 2026 - NonExistentSprint?",
        "Expected_Intent": "PRIORITY_ANALYSIS",
        "Expected_Target": "Sprint CRM Retirement Fund - 2026 - NonExistentSprint",
        "Difficulty": "Medium",
        "Ambiguity": "Clear"
    },
    {
        "Test_ID": "T60",
        "Test_Group": "G",
        "Query": "What is the weather in Sydney today?",
        "Expected_Intent": "OUT_OF_SCOPE",
        "Expected_Target": "",
        "Difficulty": "Easy",
        "Ambiguity": "Clear"
    }
]


def save_results(results: List[Dict[str, Any]], csv_path: str, xlsx_path: str):
    df = pd.DataFrame(results)
    columns = [
        "Test_ID",
        "Test_Group",
        "Query",
        "Expected_Intent",
        "Expected_Target",
        "Difficulty",
        "Ambiguity",
        "Actual_Output",
        "Execution_Status",
        "Execution_Timestamp",
        "Technical_Error",
        "Response_Time_ms",
        "Session_ID",
        "Judge_1",
        "Judge_2",
        "Judge_3",
        "Final_Label",
        "Agreement",
        "Comments"
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    df = df[columns]

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")


def main():
    print("=" * 70)
    print("STARTING EXECUTION OF 60 EXPERIMENTAL TEST CASES AGAINST CHATBOT")
    print("=" * 70)

    client = TestClient(app)
    csv_path = "chatbot_60_test_results.csv"
    xlsx_path = "chatbot_60_test_results.xlsx"
    summary_path = "execution_summary.txt"

    results = []
    total_tests = len(TEST_CASES)
    executed_count = 0
    error_count = 0
    timeout_count = 0
    response_times = []

    start_all_time = time.time()

    for idx, tc in enumerate(TEST_CASES, start=1):
        test_id = tc["Test_ID"]
        query = tc["Query"]
        group = tc["Test_Group"]

        # Session ID handling:
        # Group F (T51-T55) shares a session to support state continuity if needed.
        # Other groups receive a fresh, independent session.
        if group == "F":
            session_id = "chat_approval_eval_session"
        else:
            session_id = f"chat_{test_id}_{int(time.time() * 1000)}"

        msg_id = f"msg_{test_id}_{int(time.time() * 1000)}"
        evt_id = f"evt_{test_id}_{int(time.time() * 1000)}"

        payload = {
            "header": {
                "event_id": evt_id,
                "event_type": "im.message.receive_v1"
            },
            "event": {
                "sender": {
                    "sender_id": {"open_id": "ou_eval_user"},
                    "sender_type": "user"
                },
                "message": {
                    "message_id": msg_id,
                    "chat_id": session_id,
                    "chat_type": "p2p",
                    "message_type": "text",
                    "content": json.dumps({"text": query})
                }
            }
        }

        print(f"\n[{idx}/{total_tests}] Executing {test_id} (Group {group}): \"{query}\"")
        exec_timestamp = datetime.now().isoformat()
        t0 = time.time()

        actual_output = ""
        exec_status = "EXECUTED"
        tech_error = ""

        try:
            resp = client.post("/api/lark/webhook", json=payload)
            t1 = time.time()
            elapsed_ms = int((t1 - t0) * 1000)
            response_times.append(elapsed_ms)

            if resp.status_code == 200:
                data = resp.json()
                actual_output = data.get("reply", "")
                if not actual_output:
                    actual_output = f"Empty reply returned (status={data.get('status')})"
                executed_count += 1
                print(f"    Status: 200 OK | Time: {elapsed_ms} ms | Reply length: {len(actual_output)} chars")
            else:
                exec_status = "ERROR"
                tech_error = f"HTTP {resp.status_code}: {resp.text}"
                actual_output = f"HTTP Error {resp.status_code}"
                error_count += 1
                print(f"    Status: ERROR HTTP {resp.status_code} | Time: {elapsed_ms} ms")

        except Exception as e:
            t1 = time.time()
            elapsed_ms = int((t1 - t0) * 1000)
            response_times.append(elapsed_ms)
            exec_status = "ERROR"
            tech_error = f"{type(e).__name__}: {str(e)}"
            actual_output = f"Technical Execution Exception: {type(e).__name__}"
            error_count += 1
            print(f"    Exception: {tech_error}")

        row = {
            "Test_ID": test_id,
            "Test_Group": group,
            "Query": query,
            "Expected_Intent": tc["Expected_Intent"],
            "Expected_Target": tc.get("Expected_Target", ""),
            "Difficulty": tc.get("Difficulty", ""),
            "Ambiguity": tc.get("Ambiguity", ""),
            "Actual_Output": actual_output,
            "Execution_Status": exec_status,
            "Execution_Timestamp": exec_timestamp,
            "Technical_Error": tech_error,
            "Response_Time_ms": elapsed_ms,
            "Session_ID": session_id,
            "Judge_1": "",
            "Judge_2": "",
            "Judge_3": "",
            "Final_Label": "",
            "Agreement": "",
            "Comments": ""
        }
        results.append(row)

        # Save checkpoint after each test case to preserve progress
        save_results(results, csv_path, xlsx_path)

    total_elapsed = time.time() - start_all_time
    avg_resp_ms = sum(response_times) / len(response_times) if response_times else 0
    min_resp_ms = min(response_times) if response_times else 0
    max_resp_ms = max(response_times) if response_times else 0

    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Total Tests Executed: {total_tests}")
    print(f"Successfully Executed: {executed_count}")
    print(f"Technical Errors: {error_count}")
    print(f"Timeouts: {timeout_count}")
    print(f"Total Elapsed Time: {total_elapsed:.2f} s")
    print(f"Average Response Time: {avg_resp_ms:.1f} ms")
    print(f"Minimum Response Time: {min_resp_ms} ms")
    print(f"Maximum Response Time: {max_resp_ms} ms")

    summary_text = (
        "===============================================================================\n"
        "60 EXPERIMENTAL TEST CASES TECHNICAL EXECUTION SUMMARY\n"
        "===============================================================================\n"
        f"Project: ChatBot_DoAnTotNghiep (Australian Retirement Fund CRM Operations)\n"
        f"Execution Timestamp: {datetime.now().isoformat()}\n"
        f"Total Tests: {total_tests}\n"
        f"Executed Successfully: {executed_count}\n"
        f"Technical Errors: {error_count}\n"
        f"Timeouts: {timeout_count}\n"
        f"Total Execution Duration: {total_elapsed:.2f} seconds\n"
        f"Average Response Time: {avg_resp_ms:.2f} ms\n"
        f"Minimum Response Time: {min_resp_ms} ms\n"
        f"Maximum Response Time: {max_resp_ms} ms\n\n"
        "Output Datasets Created:\n"
        f"  - CSV: {csv_path}\n"
        f"  - Excel: {xlsx_path}\n\n"
        "Annotation Columns (Judge_1, Judge_2, Judge_3, Final_Label, Agreement, Comments):\n"
        "  - Initialized completely EMPTY as required for 3 human annotators.\n"
        "===============================================================================\n"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Saved results to {csv_path}, {xlsx_path}, and {summary_path}")


if __name__ == "__main__":
    main()
