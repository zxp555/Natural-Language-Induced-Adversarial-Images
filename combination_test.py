from allpairspy import AllPairs
from midjourney import *
from CLIP import *
from json import *

prompts = {
    "number": ["one", "two"],
    "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "humid"],
    "background": ["on the sky covered with clouds", "on the green grass field covered with flowers", "on the ground covered with snow and ice", "on the busy street", "in front of a brick wall", "inside a living room which is in a total mess", "in the dense forest", "in the rocky terrain", "under the deep sea", "on the moon", "on Mars"],
    "color": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange", "brown", "of many different colors"],
    "view angle": ["from an eye-level perspective"],
    "appearance": ["wearing a hat", "wearing a pair of glasses", "wearing clothes", "wearing a flower on the head"],
    "gesture": ["sitting", "flying", "taking a nap", "running", "playing with a ball", "chasing a butterfly", "digging a burrow", "crawling", "stretching", "barking"],
    "style": ["", "blurry,fuzzy,misty", "realistic"],
    "expression": ["happy", "sad", "angry", "worried", "depressed"]
}

# Generate all possible pairwise combinations
combinations = list(AllPairs(prompts.values()))



class PerformanceEvaluator(object):
    def __init__(self):
        pass
    def evaluate_performance(self,i):
        target_num = 0
        miss_num = 0
        success_num = 0
        image_path = i
        image = transform(Image.open(image_path)).unsqueeze(0)  
        with torch.no_grad():
            outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        probabilities = torch.softmax(outputs, dim=1)
        probabilities = probabilities.numpy()
        # predicted_class_index = torch.argmax(outputs[0]).item()        
        # probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # range_probabilities = probabilities[348:354]
        # prob = torch.sum(range_probabilities)

        if predicted_class == 1 :
            success_num += 1
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"resnet101 says it is a black person,and the probability is:{probabilities[0][1]} \n")
        # elif predicted_class == 9 :
        #     target_num += 1
        #     file_path = "result.txt"
        #     with open(file_path, "a") as file:
        #         file.write(f"resnet101 says it is a wolf,and the probability is:{probabilities[0][9]} \n")
        else :
            miss_num += 1
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"resnet101 says it is it is not a black person,and the probability is:{1-probabilities[0][1]} \n")   


        image = Image.open(image_path)
        natural_score_list = clip_ft(["a photo of an east asain","a photo of a black person"],image)
        natural_score = float(natural_score_list[0][0])
        file_path = "result.txt"
        with open(file_path, "a") as file:
            file.write(f"The image's natural score is:{natural_score/2} \n") 
            file.write(f"\n") 
        if os.path.exists(image_path) :
            os.remove(image_path)
    
        return 1-probabilities[0][1] + natural_score/2 , success_num  , target_num ,miss_num 
    

def evaluate_solution_comb(prompt: str, evaluator: PerformanceEvaluator, index):
        tot_score = 0.0
        success_number = 0
        target_number = 0
        miss_number = 0
        for i in range(2):
            out_file = f"image/{index}_{i}.png"
            task_id = ""
            image_names = None
            while (image_names == None):
                if task_id != "":
                    time.sleep(180)
                task_id = send(prompt)
                time.sleep(10)
                image_url = ""
                if task_id != "" :
                    while(image_url == ""):
                        image_url = get(task_id)
                        if(image_url == "failure"):
                            print("drawing process failed")
                            break
                        print("Still in process")
                        time.sleep(10)
                print("painting success!")
                if image_url != "" and image_url != "failure":
                    download_image(image_url,out_file)
                image_names = crop_image_to_quarters(out_file) 
            score = 0.0
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"The image index is {index},current round is {i+1}. \n")
            for image in image_names :
                score, success ,target, miss = evaluator.evaluate_performance(image)
                success_number += success
                target_number += target
                miss_number += miss
                tot_score += score
        success_rate = (target_number+miss_number) / 8
        target_rate =  target_number/8
        miss_rate = miss_number / 8
        data = {
            "index": index,
            "prompt": prompt,
            "score": tot_score,
            "success_rate":success_rate,
            "target_rate":target_rate,
            "miss_rate":miss_rate,
        }
        with open ('scores.json', 'a') as f:
            dump(data, f)
            f.write(',\n')    
        
        return tot_score,success_rate