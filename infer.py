import numpy as np
import json

def vectorize_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        sequence = list(sequence)
        results[i, sequence] = 1.
    return results

class InferModule:
    def __init__(self, model):
        self.model = model
        self.score_bias = []
        with open('./data/hobby_dataset.json', encoding='utf8') as f:
            hobby_dataset = json.load(f)
            self.position_score = hobby_dataset["list"]
            self.num2hobby = hobby_dataset["hobby_enum"]

        for idx in range(len(self.position_score)):
            if idx == 0:
                self.score_bias.append(0)
            else:
                self.score_bias.append(sum(self.position_score[:idx]))

    def start_inferring(self, user_answer):
        question_type = [0, 0, 1, 1, 2, 2, 3, 3]
        infer_score = [0, 0, 0, 0]
        u_type = ''
        for q_type, u_answer in zip(question_type, user_answer):
            if u_answer == 3:
                infer_score[q_type] += 3
            elif u_answer == 2:
                infer_score[q_type] += 2
            elif u_answer == 1:
                infer_score[q_type] += 1

        infer_score = [round(score / 2) for score in infer_score]

        if infer_score[0] >= 2: u_type += 'E'
        else: u_type += 'I'
        if infer_score[1] >= 2: u_type += 'N'
        else: u_type += 'S'
        if infer_score[2] >= 2: u_type += 'F'
        else: u_type += 'T'
        if infer_score[3] >= 2: u_type += 'J'
        else: u_type += 'P'

        X_infer = [(a+b-1) for a, b in zip(infer_score, self.score_bias)]
        X_infer = vectorize_sequences([X_infer], sum(self.position_score))
        predictions = self.model.predict(X_infer)
        hobby = []
        for pred in predictions:
            ind = np.argpartition(pred, -3)[-3:]
            ind = ind[np.argsort(pred[ind])][::-1]
            for i in ind:
                hobby.append(self.num2hobby[str(i)])
        inferringResponse = {
                "hobbyType":{
                "name":u_type
            },
            "hobbies":[
                {
                    "name":hobby[0]
                },
                {
                    "name":hobby[1]
                },
                {
                    "name":hobby[2]
                },
            ]
        }
        return inferringResponse