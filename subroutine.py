import json
import os
import base64
from openai import OpenAI

def generate_subroutine(llm_api_key, subtask_chain, image_path, prompt, subroutine_rules):
    client = OpenAI(api_key=llm_api_key)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"

    def load_chain(path: str):

        with open(path, "r", encoding="utf‑8") as fp:
            data = json.load(fp)
        return {item["subtask"]: item["parent"] for item in data["subtask_chain"]}
    
    def build_paths(graph):

        roots = [n for n, parents in graph.items() if not parents]

        paths = []

        def dfs(node, trail):
            trail.append(node)
            children = [n for n, parents in graph.items() if node in parents]
            if not children:
                paths.append(trail.copy())
            else:
                for c in children:
                    dfs(c, trail)
            trail.pop()

        for r in roots:
            dfs(r, [])
        return paths
    
    def arrows(trail):
        return " -> ".join(trail)
    
    chain_list = []
    graph = load_chain(subtask_chain)
    for trail in build_paths(graph):
        chain_list.append(arrows(trail))

    chain_text = "\n".join(f"{i+1}. {path}" for i, path in enumerate(chain_list))

    message = f"""So we have this image: <image> and also have the following input prompt:

{prompt}

So we got the following subtask chain:

{chain_text}

Note that in the subtask chain within a particular node there is a bracket which tell us about
the object from and target and if there is only from then target is not mentioned like in removal
only from object is needed no target is required while for replacement/recoloration target is
also required.
Now we have the following subroutines list for each subtask and each of the subroutines
have some observations related to them which specify under which conditions they are to
be used or not. So you need to read those subroutines and their observations then check the
corresponding object for that subtask within the image like if its Object Removal (Cat)
then check the cat in image and then from the subroutines list check that if for that particular
subtask there is any subroutine in which the observation conditions are satisfied and if so give
the list of those subroutines for that subtask and you need to do this for all subtasks in the
subtask chain.

Subroutine list and the details:

{json.dumps(subroutine_rules, indent=4)}

Example:

Suppose we have an image which has lots of objects along with a very large car which has a
background with lots of objects and also a brown wooden board with some text written on it.
Now we have a prompt that remove the car and recolor the wooden board to pink and detect
the text and get the following subtask chain:

Object Removal (Car)(1) -> Object Recoloration (Wooden Board -> Pink
Wooden Board)(2) -> Text Detection ()(3)

Now we see the subroutine list and find that for removal since the object is too big SR7 and
SR8 are not possible. Now in SR9 and SR10 we see that the ’car’ class is supported by yolo
so eventually we choose SR10 for this subtask. For recoloration we see that it has object
(text) which is imp and is involved in subsequent subtask so SR1 and SR3 aren’t possible
and we see that the color of board is light brown so light brown and pink dont have too much
difference so we choose SR2. For text detection there is not subroutine available so we leave
it like that.

So output will be:

Object Removal (Car)(1) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board)(2) : [SR2]
Text Detection ()(3) : ['None']

Now lets say the wooden board was black in color and had to be recolored to white. In this
case the SR1 and SR3 are not possible because of the text as before but now SR2 is also not
possible because the color difference is too much. So we do not choose any subroutine for
this subtask and output is as follows:

Object Removal (Car)(1) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board)(2) : ['None']
Text Detection ()(3) : ['None']

Now lets change the details further. Lets say that the wooden board does not have any text
written on it and has to be recolored from pink to yellow and the text detection subtask wasn’t
present so in this case for recoloration all subroutines are possible except SR3 bcz wooden
board isnt a class supported by yolo.

New output:

Object Removal (Car)(1) : [SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board)(2) : [SR1, SR2]

Now lets change it a bit assume that all conditions are as original but the car is small and
behind the car only some walls, grass, etc are present some basic stuff and not a lot of objects
like occluded people, cats, etc so in this case we will choose SR8 and SR10 for it as it is not
too plain that SR10 cannot be used and it is not way too complex that SR8 cannot be used.

New output:

Object Removal (Car)(1) : [SR8, SR10]
Object Recoloration (Wooden Board -> Pink Wooden Board)(2) : [SR2]
Text Detection ()(3) : ['None']

Now you need to do the same things for the current case where the input prompt is : {prompt}
Subtask chain: {chain_text}

Also multiple options are possible if they satisfy all the conditions it is not necessary that only
one is chosen and it is also possible that no subroutine fulfills all conditions so in that case
choose None so that we can do A∗ search and find the correct output. Also keep in mind that
only look at the details relevant say you need to check subroutine for some object which is to
be removed and for some activation condition you need to see if the background is busy or
plain, etc so you only see the background relevant like near that object and not for the entire
image.

So you need to extract all relevant details related to all relevant objects from the image given
to you then check the subroutine list if anyone matches and give the output."""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message + "\n\nYou must respond in a valid JSON format. Do not include any extra text before or after the JSON output."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    content = response.choices[0].message.content
    if content.startswith("```json"):
        content = content[7:-3]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to parse response:")
        print("Raw content:", content)
        raise ValueError(f"Invalid JSON response: {str(e)}") from e
    
def main():
    llm_api_key = os.getenv("OPENAI_API_KEY")
    subtask_chain = "Sub.json"           # Replace with your subtask chain JSON file path
    image_path = "image.jpg"            # Replace with your image path
    prompt = "Remove the cat and recolor the wooden board to pink."

    with open("subroutine_rule_table.json", 'r') as f:
        subroutine_rules = json.load(f)

    result = generate_subroutine(llm_api_key, subtask_chain, image_path, prompt, subroutine_rules)
    print(json.dumps(result, indent=4))

# # Uncomment the following line to run the script directly
# if __name__ == "__main__":
#     main()
