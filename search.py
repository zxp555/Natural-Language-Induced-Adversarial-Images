from genetic_algorithm import *
from combination_test import *

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default="gene", type=str)
    parser.add_argument('--target', default="fire truck", type=str)
    parser.add_argument('--task', default="animal", type=str)
    parser.add_argument('--targetlabel', default=0, type=int)
    parser.add_argument('--target_score', default=20, type=int)
    parser.add_argument('--max_generations', default=50, type=int)
    parser.add_argument('--selectclass', default=[407,408,468,511,555,569,573,609,654,751], type=int, nargs='+')
    args = parser.parse_args()
    if args.algo == "gene":
        print(args.selectclass)
        print(type(args.selectclass))
        optimizer = GeneticAlgorithmPromptOptimization(args.target,args.target_score, args.max_generations,args.task,args.targetlabel,args.selectclass)
        best_prompt = optimizer.optimize_prompt()

    elif args.algo == "comb":
        fitnesses = []
        success_rates = []
        for i,combination in enumerate(combinations):
            prompt = ""
            if combination[7] == "":
                prompt = f"{combination[0]} {combination[3]} {args.target} {combination[5]} is {combination[6]} {combination[2]} on a {combination[1]} day, the {args.target} occupies the main part in this scene, viewed {combination[4]}  --q .25"
            else :
                prompt = f"{combination[0]} {combination[3]} {args.target} {combination[5]} is {combination[6]} {combination[2]} on a {combination[1]} day, the {args.target} occupies the main part in this scene,in a {combination[7]} style, viewed {combination[4]}  --q .25"  
            evaluator = PerformanceEvaluator()
            fitness,success_rate = evaluate_solution_comb(prompt,evaluator,i)
            if success_rate == 1.0:
                sys.exit()
            fitnesses.append(fitness)
            success_rates.append(success_rate)
        total_fitness = 0.0
        total_success_rate =0.0
        for i in fitnesses :
            total_fitness += i
            total_fitness = total_fitness / len(fitnesses)
        for i in success_rates :
            total_success_rate += i
            total_success_rate = total_success_rate / len(success_rates)

        file_path = "avg_fitness.txt"
        with open(file_path, "a") as file:
            file.write(f"avg fitness is {total_fitness}. \n")
            file.write(f"avg success rate is {total_success_rate}. \n")

    elif args.algo == "random":
        prompt_space =  {
        "number":["one"],
        "weather": ["sunny", "rainy", "cloudy", "snowy", "windy", "foggy", "stormy", "humid"],
        "background": ["on the sky covered with clouds", "on the green grass field covered with flowers", "on the ground covered with snow and ice", "on the busy street", "in front of a brick wall", "inside a living room which is in a total mess", "in the dense forest", "in the rocky terrain","under the deep sea","on the moon", "on Mars"],
        "color": [""],
        "view angle": ["from an eye-level perspective"],
        "appearance":["wearing a hat","wearing a pair of glasses", "wearing formal suits", "wearing casual wear","wearing traditional attires", "wearing athletic outfits","with long hair" , "with short hair","with curly hair" ,"wearing a flower on the head","with tatoo on the face","wearing necklaces", "wearing earrings", "wearing bracelets"],
        "gesture": ["sitting", "smoking", "taking a nap", "running", "playing with a ball", "chasing a butterfly", "digging a burrow", "crawling", "stretching","studying","exercising","working"],
        "style": ["","blurry,fuzzy,misty", "realistic"],
        "expression":["happy","sad","angry","worried","depressed","overwhelmed"]}
        success_number = 0
        for i in range(0,10):
            selected_attributes = {}
            for key, values in prompt_space.items():
                selected_attributes[key] = random.choice(values)

            prompt = f"{selected_attributes['number']} {selected_attributes['color']}{selected_attributes['expression']}  {args.target} {selected_attributes['appearance']} is {selected_attributes['gesture']} {selected_attributes['background']} on a {selected_attributes['weather']} day, the {args.target} occupies the main part in this scene, viewed {selected_attributes['view angle']}  --q .25"
            evaluator = PerformanceEvaluator()
            fitness,success_rate = evaluate_solution_comb(prompt,evaluator,i)
            success_number += success_rate * 8
        print(f"success number: {success_number}")
