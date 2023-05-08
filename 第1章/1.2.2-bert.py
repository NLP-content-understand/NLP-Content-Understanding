from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

batch_sentence = [
    '这是第一个句子',
    '这是第二个句子哈哈哈',
    '我是第三句'
]

input_id = tokenizer(batch_sentence, padding=True, truncation=True, max_length=10, return_tensors='pt')
print(input_id)
print(input_id['input_ids'].shape)

output = model(input_id['input_ids'])
print(output[0].shape)
print(output[1].shape)