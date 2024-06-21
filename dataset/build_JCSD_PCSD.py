import json

root_path = "cleanJCSD4Sit"
type = 'tlcodesum.'
total_nl = open(root_path + '/total_nl.txt', "w", encoding="utf-8")

def read_examples(filename, root_path):
    raw_code = open(root_path + '/' + i + '/code', "w", encoding="utf-8")
    raw_nl = open(root_path + '/' + i + '/nl', "w", encoding="utf-8")
    ref = open(root_path + '/' + i + '/ref.txt', "w", encoding="utf-8")
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # if idx>=3:
            #     break
            line=line.strip()
            js=json.loads(line)

            id = js['id']
            code = js['code'].replace('\n', ' ')
            nl = js['comment']

            raw_code.write(code + '\n')
            raw_nl.write(nl + '\n')
            ref.write(str(idx+1) + '\t' + nl + '\n')
            total_nl.write(nl + '\n')

    raw_nl.close()
    raw_code.close()
    ref.close()



if __name__ == "__main__":

    for i in ['test', 'train', 'valid']:
        filename = root_path + '/' + i + '/' + type + i
        # filename = root_path + '/' + i + '/test.json'
        read_examples(filename, root_path)

    total_nl.close()