import argparse
import json
import os
from tqdm import tqdm
import openai
import time

NUM_SECONDS_TO_SLEEP = 2
# base_url = "xxx"
# api_key = "xxx"
# CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url)


class Chat:
    def __init__(self, model="", timeout_sec=20, openai_apikey='', apibase=''):
        self.model = model
        self.timeout = timeout_sec
        self.client = openai.AzureOpenAI(
            azure_endpoint=apibase,
            api_version="2023-03-15-preview",
            api_key=openai_apikey
        )

    def chat_completion(self, messages, temperature=0, top_p=1, max_tokens=512,
                        presence_penalty=0, frequency_penalty=0):

        response = self.client.chat.completions.create(
            extra_headers={"X-TT-LOGID": "xxx"},
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            timeout=self.timeout
        )

        return response
    

def get_eval(client, content: str, max_tokens: int):
    for i in range(4):
        try:
            response = client.chat_completion(messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }], temperature=0.2, max_tokens=max_tokens)
            # response = client.chat.completions.create(
            #     model='gpt-4o-2024-11-20',
            #     messages=[{
            #         'role': 'system',
            #         'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
            #     }, {
            #         'role': 'user',
            #         'content': content,
            #     }],
            #     temperature=0.2,  # TODO: figure out which temperature is best for evaluation
            #     max_tokens=max_tokens,
            # )
            break
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP*(2**i))

    return response.choices[0].message.content


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split(' ')
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('score parsing error', review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print('score parsing error', review)
        return [-1, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ChatGPT-based QA evaluation.')
    parser.add_argument('-q', '--question')
    parser.add_argument('-c', '--context')
    parser.add_argument('-a', '--answer-list', nargs='+', default=[])
    parser.add_argument('-r', '--rule')
    parser.add_argument('-o', '--output')
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--api-base", type=str, default='')
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    parser.add_argument("--model", default="gpt-4o-2024-05-13")
    args = parser.parse_args()

    client = Chat(model=args.model, timeout_sec=20, openai_apikey=args.api_key, apibase=args.api_base)
    f_q = open(os.path.expanduser(args.question))
    f_ans1 = open(os.path.expanduser(args.answer_list[0]))
    f_ans2 = open(os.path.expanduser(args.answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context))]
    image_to_context = {context['image']: context for context in context_list}

    handles = []
    idx = 0
    for ques_js, ans1_js, ans2_js in tqdm(zip(f_q, f_ans1, f_ans2)):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)
        
        assert ques["question_id"]==ans1["question_id"] and ques["question_id"]==ans2["question_id"]

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (f'[Context]\n{cap_str}\n\n'
                   f'[Question]\n{ques["text"]}\n\n'
                   f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
                   f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
                   f'[System]\n{prompt}\n\n')
        cur_js = {
            'id': idx+1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category
        }
        if idx >= len(cur_reviews):
            review = get_eval(client, content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.', flush=True)
        idx += 1

    review_file.close()
