import os
import json

def read_mm_safebench(dataset_path, img_type="SD_TYPO",query_type = "Rephrased Question(SD)"):

    processed_questions_dir = os.path.join(dataset_path, "processed_questions")
    img_dir = os.path.join(dataset_path, "imgs")
    data_list = []


    allowed_scenarios = {
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "06-Fraud",
        "09-Privacy_Violence",
    }


    for json_file in os.listdir(processed_questions_dir):
        if not json_file.endswith(".json"):
            continue

        scenario = os.path.splitext(json_file)[0]
        
        if scenario not in allowed_scenarios:
            continue  
        json_path = os.path.join(processed_questions_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            questions = json.load(f)
        
        for qid, content in questions.items():
      
            query_text = content[query_type]
            harm_insturction = content["Changed Question"]
         
            image_path = os.path.join(img_dir, scenario, img_type, f"{qid}.jpg")
            
            data_list.append({
                "scenario": scenario,
                "question_id": qid,
                "ori_harm_query":harm_insturction,
                "query": query_text,
                "image": image_path
            })

    return data_list
