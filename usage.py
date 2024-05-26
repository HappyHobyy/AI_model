from infer import InferModule
from keras import models
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = models.load_model('./hobby_model.keras')
IM = InferModule(model)

if __name__ == '__main__':
    # 사용자의 문항별 답변 항목
    inferringResponse = IM.start_inferring([0,2,3,1,2,3,2,1])
    print(inferringResponse)