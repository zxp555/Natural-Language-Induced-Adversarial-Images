import openai
import re
import json



def query_gpt4(prompt):
    client = openai.OpenAI(api_key='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')

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a creative and knowledgeable assistant. "},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
    )

    return completion.choices[0].message.content


def extract_prompt_template(response):
    match = re.search(r'f".*?"', response, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
    
def extract_and_modify_dict(s):
    try:
        extracted = re.search(r'\{(.+?)\}', s).group(0)
        d = eval(extracted)
        d.pop('type', None)
        return d
    except Exception as e:
        return None

def wordspace_generation(task:str):
    prompt = f'Now I need you to help construct a word space for the given task {task}.'
    fewshot_examples ="Here are two examples of the constructed word space. First is for animal classification.The word space is :\n \
    {'number': ['one','two'],'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'humid'],'background': ['on the sky covered with clouds', 'on the green grass field covered with flowers', 'on the ground covered with snow and ice', 'on the busy street', 'in front of a brick wall', 'inside a living room which is in a total mess', 'in the dense forest', 'in the rocky terrain', 'under the deep sea', 'on the moon', 'on Mars'],\
    'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange', 'brown', 'of many different colors'],'view angle': ['from an eye-level perspective'],'appearance': ['wearing a hat', 'wearing a pair of glasses', 'wearing clothes', 'wearing a flower on the head'],\
    'gesture': ['sitting', 'flying', 'taking a nap', 'running', 'playing with a ball', 'chasing a butterfly', 'digging a burrow', 'crawling', 'stretching', 'barking','standing'],'style': ['', 'blurry,fuzzy,misty', 'realistic'],'expression': ['happy', 'sad', 'angry', 'worried', 'depressed']'} \n \
     And the second example is for human race classification. The word space is: \n \
    {'number':['one'],'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'humid'],'background': ['on the sky covered with clouds', 'on the green grass field covered with flowers', 'on the ground covered with snow and ice', 'on the busy street', 'in front of a brick wall', 'inside a living room which is in a total mess', 'in the dense forest', 'in the rocky terrain','under the deep sea','on the moon', 'on Mars'], \
    color': [],view angle': ['from an eye-level perspective'],'appearance':['wearing a hat','wearing a pair of glasses', 'wearing formal suits', 'wearing casual wear','wearing traditional attires', 'wearing athletic outfits','with long hair' , 'with short hair','with curly hair' ,'wearing a flower on the head','with tatoo on the face','wearing necklaces', 'wearing earrings', 'wearing bracelets'],'gesture': ['sitting', 'smoking', 'taking a nap', 'running', 'playing with a ball', 'chasing a butterfly', 'digging a burrow', 'crawling', 'stretching','studying','exercising','working'],\
    'style': ['','blurry,fuzzy,misty', 'realistic'],'expression':['happy','sad','angry','worried','depressed','overwhelmed']}' \n \
    "
    prompt += fewshot_examples
    prompt += "Both word spaces have descriptions for three parts: the object attribute(like number, gesture,appearance and so on), background information(weather,background,time of day and so on),image feature(view angle,style and so on).\
        When constructing the word space, you can also think from these three perspective. But pay attention that each task may have its unique feature, like humans may wear various clothes but their skin colors are limited; and animals may have unique behaviours like barking or flying.\
            When constructing word space,you must take these characteristics of different tasks into consideration and construct the most suitable word space."
    prompt += f"Do remember that your constructed word space should have the same attribute:number,weather,background,color,view angle,gesture,style,appearance,expression. If you find one attribute is not suitable for the task,\
        you should keep the key and give a list [''] as its value. Now construct the word space for the task of {task} classification. Your answer must be in the form of a python dictionary like the example above. Do not include any words other than the dictionary!"
    print(prompt)
    word_space = None
    while word_space is None:
        # response = query_gpt4(prompt)
        response = "{'number': ['one', 'two', 'three', 'multiple'], 'weather': ['sunny', 'rainy', 'cloudy', 'snowy', 'windy', 'foggy', 'stormy', 'humid'], 'background': ['on the highway', 'in a parking lot', 'on a city street', 'in a garage', 'on a race track', 'in a rural area', 'near a body of water', 'in a desert', 'in a forest', 'on a bridge'], 'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'silver', 'gray', 'orange', 'brown'], 'view angle': ['from the front', 'from the side', 'from the back', 'from above'], 'gesture': [''], 'style': ['', 'blurry', 'fuzzy', 'misty', 'realistic'], 'appearance': ['with headlights on', 'with doors open', 'with a spoiler', 'with a sunroof', 'with tinted windows'], 'expression': ['']}"
        word_space = extract_and_modify_dict(response)

    return word_space

