import json

# java
# root_path = "cleanJCSD4Sit"
# type = 'tlcodesum.'
# skip = ['87140', '87138', '87137', '87136']

root_path = "cleanPCSD4Sit"
type = 'pcsd.'
skip = []
# skip = ['87140', '87138', '87137', '87136']

def read_examples(filename, root_path):
    raw_code = open(root_path + '/' + i + '.token.code', "w", encoding="utf-8")
    raw_nl = open(root_path + '/' + i + '.token.nl', "w", encoding="utf-8")
    raw_guid = open(root_path + '/' + i + '.token.guid', "w", encoding="utf-8")
    ref = open(root_path + '/ref.txt', "w", encoding="utf-8")
    with open(filename,encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # if idx>=3:
            #     break
            line=line.strip()
            js=json.loads(line)

            id = js['id']
            if id in skip:
                continue

            code = js['code'].replace('\n', ' ')
            nl = js['comment']

            raw_code.write(code + '\n')
            raw_nl.write(nl + '\n')
            raw_guid.write(id + '\n')
            ref.write(str(idx+1) + '\t' + nl + '\n')

    raw_nl.close()
    raw_code.close()
    ref.close()



if __name__ == "__main__":

    for i in ['test', 'train', 'valid']:
        filename = root_path + '/' + type + i
        # filename = root_path + '/' + i + '/test.json'
        read_examples(filename, root_path)
