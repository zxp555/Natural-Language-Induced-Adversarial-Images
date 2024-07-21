import PIL
from tqdm import tqdm
from midjourney import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
import subprocess
import random
import sys
import time
from json import *
from CLIP import *
from wordspace import *
# from flava import *

class PerformanceEvaluator(object):
    def __init__(self):
        pass
    def evaluate_performance(self,i,targetlabel,selectclass):
        target_num = 0
        miss_num = 0
        success_num = 0
        image_path = i

        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)


        
        selected_classes = selectclass # ambulance, amphibious vehicle, taxi,convertible car, fire truck, garbage truck, go-kart, jeep, minibus, race car
        selected_probs = probabilities[0, selected_classes]
        normalized_probs = selected_probs / selected_probs.sum()
        predicted_class = selected_classes[normalized_probs.argmax().item()]


        target_index = targetlabel

        # _, predicted = torch.max(outputs, 1)
        # predicted_class = predicted.item()
        # probabilities = torch.softmax(outputs, dim=1)
        # probabilities = probabilities.numpy()
        # predicted_class_index = torch.argmax(outputs[0]).item()        
        # probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        # range_probabilities = probabilities[348:354]
        # prob = torch.sum(range_probabilities)

        # image = Image.open(image_path)
        # prompt_list = ["a photo of a butterfly","a photo of a cat","a photo of a chicken","a photo of a cow","a photo of a dog","a photo of an elephant","a photo of a horse","a photo of a sheep", "a photo of a squirrel","a photo of a spider"] 
        # prob_list = clip_ft(prompt_list,image)
        # target_prob = prob_list[0, 0]
        # target_prob = target_prob.item()
        # print(target_prob)
        # predicted_class = torch.argmax(prob_list)
        # predicted_class = predicted_class.item()
        # print(predicted_class)
        if predicted_class == selected_classes[target_index] :
            success_num += 1
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"resnet says it is a convertible car,and the probability is:{normalized_probs[target_index].item()} \n")
        # elif predicted_class == 9 :
        #     target_num += 1
        #     file_path = "result.txt"
        #     with open(file_path, "a") as file:
        #         file.write(f"resnet101 says it is a wolf,and the probability is:{probabilities[0][9]} \n")
        else :
            miss_num += 1
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"resnet says it is it is not a convertible car,and the probability is:{1-normalized_probs[target_index].item()} \n")   

        image = Image.open(image_path)
        natural_score_list = clip_ft(['a photo of a fire truck','a photo of a race car'], image)
        natural_score = float(natural_score_list[0][0])

        file_path = "result.txt"
        with open(file_path, "a") as file:
            file.write(f"The image's natural score is:{natural_score/2} \n") 
            file.write(f"\n") 
        if os.path.exists(image_path) :
            os.remove(image_path)
    
        return 1- normalized_probs[target_index].item() + natural_score/2 , success_num  , target_num ,miss_num 


class PromptDB(object):
    def __init__(self,task):
        # prompts = {
        # "number":["one"],
        # "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "humid"],
        # "background": ["on the sky covered with clouds", "on the green grass field covered with flowers", "on the ground covered with snow and ice", "on the busy street", "in front of a brick wall", "inside a living room which is in a total mess", "in the dense forest", "in the rocky terrain","under the deep sea","on the moon", "on Mars"],
        # "color": [""],
        # "view angle": ["from an eye-level perspective"],
        # "appearance":["wearing a hat","wearing a pair of glasses", "wearing formal suits", "wearing casual wear","wearing traditional attires", "wearing athletic outfits","with long hair" , "with short hair","with curly hair" ,"wearing a flower on the head","with tatoo on the face","wearing necklaces", "wearing earrings", "wearing bracelets"],
        # "gesture": ["sitting", "smoking", "taking a nap", "running", "playing with a ball", "chasing a butterfly", "digging a burrow", "crawling", "stretching","studying","exercising","working"],
        # "style": ["","blurry,fuzzy,misty", "realistic"],
        # "expression":["happy","sad","angry","worried","depressed","overwhelmed"]}

        prompts = {'number': ['one', 'two', 'three', 'multiple'], 
    'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'humid'], 
    'background': ['on the highway', 'in a parking lot', 'on a city street', 'in a garage', 'on a race track', 'in a rural area', 'near a body of water', 'in a desert', 'in a forest', 'on a bridge'], 
    'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'silver', 'gray', 'orange', 'brown'],
    'view angle': ['from the front', 'from the side', 'from the back', 'from above'], 
    'gesture': ['parking', 'galloping at full speed', 'reversing', 'turning', 'drifting', 'stopping'],
    'style': ['','blurry', 'fuzzy', 'misty', 'realistic', 'cartoonish', 'sketchy', 'vintage', 'modern'],
    'appearance': ['with headlights on', 'with doors open', 'with a spoiler', 'with a sunroof', 'with tinted windows', 'with a roof rack', 'with decals', 'with a trailer', 'with a roof box', 'with custom wheels'],
    'expression': ['old', 'new', 'worn', 'broken', 'shiny', 'rusty', 'dirty', 'clean']
}

        if task != 'animal':
            prompts = wordspace_generation


#         prompts = {
#     "number": ["one", "two"],
#     "weather": ["rainy", "cloudy", "snowy", "windy", "foggy"],
#     "background": ["on the sky covered with clouds", "on the busy street","inside a living room which is in a total mess", "on the moon", "on Mars"],
#     "color": ["red", "blue", "green", "white", "purple", "orange"],
#     "view angle": ["from an eye-level perspective"],
#     "appearance": ["wearing a hat", "wearing a pair of glasses", "wearing clothes", "wearing a flower on the head"],
#     "gesture": ["flying", "taking a nap", "running", "playing with a ball", "chasing a butterfly", "digging a burrow"],
#     "style": ["", "realistic"],
#     "expression": ["happy", "sad", "angry", "worried", "depressed"]
# }
        self.number = prompts["number"]
        self.weather = prompts["weather"]
        self.background = prompts["background"]
        self.color = prompts["color"]
        self.viewangle = prompts["view angle"]
        self.gesture = prompts["gesture"]
        self.style = prompts["style"]
        self.appearance = prompts["appearance"]
        self.expression = prompts["expression"]
    

class Individual(object):
    def __init__(self, **kwargs):
        self.number = kwargs.get("number")
        self.weather = kwargs.get("weather")
        self.background = kwargs.get("background")
        self.color = kwargs.get("color")
        self.viewangle = kwargs.get("view angle")
        self.gesture = kwargs.get("gesture")
        self.style = kwargs.get("style")
        self.appearance = kwargs.get("appearance")
        self.expression = kwargs.get("expression")
        self.fitness_score = 0.0
        self.success_rate = 0.0
        self.target_rate = 0.0
        self.miss_rate = 0.0

    def random(self, promptDB):
        self.number = random.choice(promptDB.number)
        self.weather = random.choice(promptDB.weather)
        self.background = random.choice(promptDB.background)
        self.color = random.choice(promptDB.color)
        self.viewangle = random.choice(promptDB.viewangle)
        self.gesture = random.choice(promptDB.gesture)
        self.style = random.choice(promptDB.style)
        self.appearance = random.choice(promptDB.appearance)
        self.expression = random.choice(promptDB.expression)
        self.fitness_score = 0.0
        self.success_rate = 0.0
        return self

    def prompt(self,target_word):
        return f"{self.number} {self.color} {self.expression} {target_word} {self.appearance} is {self.gesture} {self.background} on a {self.weather} day,the {target_word} faces forward, the {target_word}  occupies the main part in this scene,in a {self.style} style  --q .25"
    def __lt__(self, other):
        return self.fitness_score < other.fitness_score

    def __gt__(self, other):
        return self.fitness_score > other.fitness_score

    def __le__(self, other):
        return self.fitness_score <= other.fitness_score

    def __ge__(self, other):
        return self.fitness_score >= other.fitness_score
        
class Fitness(object):
    def __init__(self,targetword):
        self.targetword = targetword
    def compute_fitness(self, individual: Individual, evaluator: PerformanceEvaluator, step, index,targetlabel,selectclass):
        tot_score = 0.0
        success_number = 0
        target_number = 0
        miss_number = 0
        for i in range(2):
            out_file = f"image/{step}_{index}_{i}.png"
            task_id = ""
            image_names = None
            while (image_names == None):
                if task_id != "":
                    time.sleep(180)
                task_id = send(individual.prompt(self.targetword))
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
                # print(image_url)
                if image_url != "" and image_url != "failure":
                    download_image(image_url,out_file)
                image_names = crop_image_to_quarters(out_file) 
            score = 0.0
            file_path = "result.txt"
            with open(file_path, "a") as file:
                file.write(f"The image index is {index},current round is {i+1}. \n")
            for image in image_names :
                score, success ,target, miss = evaluator.evaluate_performance(image,targetlabel,selectclass)
                success_number += success
                target_number += target
                miss_number += miss
                tot_score += score
        success_rate = (target_number+miss_number) / 8
        target_rate =  target_number/8
        miss_rate = miss_number / 8
        data = {
            "step": step,
            "index": index,
            "prompt": individual.prompt(self.targetword),
            "score": tot_score,
            "success_rate":success_rate,
            "target_rate":target_rate,
            "miss_rate":miss_rate,
        }
        with open ('scores.json', 'a') as f:
            dump(data, f)
            f.write(',\n')    
        
        return tot_score,success_rate,target_rate,miss_rate

class GeneticAlgorithm(object):
    def __init__(self,target_word,task,targetlabel,selectclass):
        self.targetword = target_word
        self.evaluator = PerformanceEvaluator()
        self.population_size = 20
        self.mutation_rate = 0.02
        self.elite_rate = 0.1
        self.population = []
        self.DB = PromptDB(task)
        self.fitness = Fitness(self.targetword) 
        self.current_step = 0
        
    def initialize_population(self):
        individual1 = Individual(
            number="one", 
            weather="rainy", 
            background="inside a living room which is in a total mess", 
            color="blue", 
            viewangle="from an eye-level perspective", 
            gesture="sitting", 
            style="realistic", 
            appearance="wearing clothes", 
            expression="happy"
        )

        individual2 = Individual(
            number="two", 
            weather="foggy", 
            background="on the moon", 
            color="white", 
            viewangle="from an eye-level perspective", 
            gesture="running", 
            style="realistic", 
            appearance="wearing a flower on the head", 
            expression="worried"
        )


        self.population.append(individual1)
        self.population.append(individual2)

        for _ in range(self.population_size):
            individual = Individual().random(self.DB)
            self.population.append(individual)

    def compute_all_fitness(self,targetlabel,selectclass):
        total_fitness = 0
        total_success_rate = 0.0
        total_target_rate = 0.0
        total_miss_rate = 0.0
        for index, individual in tqdm(enumerate(self.population), total=len(self.population)):
            # print(self.current_step, index, self.population_size, len(self.population))
            self.population[index].fitness_score,self.population[index].success_rate,self.population[index].target_rate,self.population[index].miss_rate = self.fitness.compute_fitness(individual, self.evaluator, self.current_step, index,targetlabel,selectclass)
            # if individual.success_rate == 1.0:
            #     print("Attack success!")
            #     sys.exit()
            total_fitness += individual.fitness_score
            total_success_rate += individual.success_rate
            total_target_rate += individual.target_rate
            total_miss_rate += individual.miss_rate
        file_path = "avg_fitness.txt"
        with open(file_path, "a") as file:
            file.write(f"avg fitness is {total_fitness / self.population_size}. \n")
            file.write(f"total attack number is {self.population_size * 8},total success number is {round(total_success_rate * 8)} \n")
            file.write(f"avg success rate is {total_success_rate / self.population_size}. \n")
            file.write(f"avg target rate is {total_target_rate / self.population_size}. \n")
            file.write(f"avg miss rate is {total_miss_rate / self.population_size}. \n")
            file.write(f"\n")

    def crossover(self, parent1, parent2):
        child = Individual()
        child.number = parent1.number if random.random() < 0.5 else parent2.number
        child.weather = parent1.weather if random.random() < 0.5 else parent2.weather
        child.background = parent1.background if random.random() < 0.5 else parent2.background
        child.color = parent1.color if random.random() < 0.5 else parent2.color
        child.viewangle = parent1.viewangle if random.random() < 0.5 else parent2.viewangle
        child.gesture = parent1.gesture if random.random() < 0.5 else parent2.gesture
        child.style = parent1.style if random.random() < 0.5 else parent2.style
        child.appearance = parent1.appearance if random.random() < 0.5 else parent2.appearance
        child.expression = parent1.expression if random.random() < 0.5 else parent2.expression
        return child
    
    def mutate(self, child):
        if random.random() < self.mutation_rate:
            child.number = random.choice(self.DB.number)
        if random.random() < self.mutation_rate:
            child.weather = random.choice(self.DB.weather)
        if random.random() < self.mutation_rate:
            child.background = random.choice(self.DB.background)
        if random.random() < self.mutation_rate:
            child.appearance = random.choice(self.DB.appearance)
        if random.random() < self.mutation_rate:
            child.color = random.choice(self.DB.color)
        if random.random() < self.mutation_rate:
            child.expression = random.choice(self.DB.expression)
        if random.random() < self.mutation_rate:
            child.viewangle = random.choice(self.DB.viewangle)
        if random.random() < self.mutation_rate:
            child.gesture = random.choice(self.DB.gesture)
        if random.random() < self.mutation_rate:
            child.style = random.choice(self.DB.style)
        return child


    def selection(self,targetlabel,selectclass):
        self.compute_all_fitness(targetlabel,selectclass)
        self.population.sort(reverse = True)
        elite_size = int(self.elite_rate * self.population_size)
        elites = self.population[:elite_size]
        fitness_sum = sum([individual.fitness_score for individual in self.population])
        selected_individuals = []
        while len(selected_individuals) < self.population_size - elite_size:
            pick = random.uniform(0, fitness_sum)
            current = 0
            for individual in self.population:
                current += individual.fitness_score
                if current > pick:
                    selected_individuals.append(individual)
                    break
        self.population = elites + selected_individuals

    def generate(self):
        new_population = []
        total_fitness = sum(individual.fitness_score for individual in self.population)
        if total_fitness == 0:
            return

        # # Find the individual with the lowest fitness score
        # lowest_individual = min(self.population, key=lambda ind: ind.fitness_score)

        # # Select two random attributes from the specified list
        # attributes_to_modify = random.sample(["gesture", "appearance", "color", "background", "weather"], 2)

        # # Remove the value of these attributes from the prompt space
        # for attribute in attributes_to_modify:
        #     attribute_list = getattr(self.DB, attribute)
        #     if getattr(lowest_individual, attribute) in attribute_list and len(attribute_list) > 3:
        #         attribute_list.remove(getattr(lowest_individual, attribute))

        # Normal genetic algorithm process
        while len(new_population) < self.population_size:
            parent1 = self.roulette_wheel_selection(total_fitness)
            parent2 = self.roulette_wheel_selection(total_fitness)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population


    def roulette_wheel_selection(self, total_fitness):
        selection_point = random.uniform(0, total_fitness)
        current = 0
        for individual in self.population:
            current += individual.fitness_score
            if current >= selection_point:
                return individual


    def step(self,targetlabel,selectclass):
        self.generate()
        self.selection(targetlabel,selectclass)
        self.current_step += 1


class GeneticAlgorithmPromptOptimization(object):
    def __init__(self, target_word,target_score=0.0, max_generations=50,task = 'animal',targetlabel=0,selectclass=[407,408,468,511,555,569,573,609,654,751]):
        self.gene = GeneticAlgorithm(target_word,task,targetlabel,selectclass)
        self.target_score = target_score
        self.max_generations = max_generations
        self.targetlabel = targetlabel
        self.selectclass = selectclass


    def optimize_prompt(self):
        self.gene.initialize_population()
        while self.gene.current_step< self.max_generations:
            self.gene.step(self.targetlabel,self.selectclass)

            best_individual = max(self.gene.population, key=lambda ind: ind.fitness_score)
            print(f"Generation {self.gene.current_step}: Best Score = {best_individual.fitness_score:.4f}")
            if best_individual.fitness_score >= self.target_score:
                print("Target score reached or exceeded. Optimization stopped.")
                break
        
        best_individual = max(self.gene.population, key=lambda ind: ind.fitness_score)
        best_prompt = best_individual.prompt(self.gene.targetword)
        best_score = best_individual.fitness_score

        print("Optimization completed.")
        print(f"Best Prompt: {best_prompt}")
        print(f"Best Score: {best_score:.4f}")

        return best_prompt

