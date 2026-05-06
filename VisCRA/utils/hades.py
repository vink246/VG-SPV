import os
import json

def process_hades_data(hades_root):
    result = []




    instructions_dir = os.path.join(hades_root, 'instructions')
    for json_file in os.listdir(instructions_dir):
        if not json_file.endswith('.json'):
            continue
            

        scenario = os.path.splitext(json_file)[0]

        
        json_path = os.path.join(instructions_dir, json_file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:

            img_folder = os.path.join(
                hades_root,
                'optimized_SD_images',
                scenario,
                str(item["id"])
            )
            img_path = os.path.join(img_folder, "concat_opt_1.jpg")
            

            category = item["category"]
            keywords = item["keywords"]
            
            if category == 'behavior':
                mask = 'conduct the behavior in the image on'
            else:
                mask = f'the {category} in the image'
            
            query = item["instruction"].replace(keywords, mask)
            

            result.append({
                "scenario": scenario,
                "question_id": item["id"],
                "keywords": keywords,
                "image": img_path.replace(os.sep, '/'),  
                "ori_harm_query": item["instruction"],
                "query": query
            })
    
    return result
