import jieba
import random


def clean_train_data(train_path, out_path):
    data = []
    with open(train_path, 'r') as f:
        for line in f:
            ls = line.split('\t')
            idx = ls[0]
            typ = ls[1]
            comment = " ".join(ls[2:len(ls) - 1])
            label = ls[-1].strip()
            data.append([idx, typ, comment, label])

    with open(out_path, 'a') as f:
        for each in data:
            f.write("\t".join(each) + "\n")

    print("完成训练集清洗，训练集总数据：%d" % len(data))


def clean_test_data(test_path, out_path):
    data = []
    with open(test_path, 'r') as f:
        for line in f:
            ls = line.split('\t')
            idx = ls[0]
            typ = ls[1]
            comment = " ".join(ls[2:]).strip()
            data.append([idx, typ, comment])

    with open(out_path, 'a') as f:
        for each in data:
            f.write("\t".join(each) + "\n")
    print("完成测试集清洗，测试集总数据：%d" % len(data))


def seg(path, out_path, isTraining, stop_words):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if isTraining:
                idx, typ, comment, label = line.strip().split('\t')
                seg_list = jieba.cut(comment)
                segs = []
                for w in seg_list:
                    if w not in stop_words:
                        segs.append(w)
                seg = " ".join(segs)
                data.append([idx, typ, seg, label])
            else:
                try:
                    idx, typ, comment = line.split('\t')
                    seg_list = jieba.cut(comment.strip())
                    segs = []
                    for w in seg_list:
                        if w not in stop_words:
                            segs.append(w)
                    seg = " ".join(segs)
                    data.append([idx, typ, seg])
                except ValueError:
                    print(line)

    with open(out_path, 'a') as f:
        for each in data:
            f.write("\t".join(each) + "\n")
    if isTraining:
        print("完成训练集分词: %d" % len(data))
    else:
        print("完成测试集分词: %d" % len(data))


def fit_fasttext(path, out_path, isTraining):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if isTraining:
                idx, typ, comment, label = line.strip().split('\t')
                data.append([comment, label])
            else:
                idx, typ, comment = line.split('\t')
                data.append(comment.strip())

    with open(out_path, 'a') as f:
        if isTraining:
            for each in data:
                comment, label = each
                f.write(comment + '\t__label__' + label + '\n')
        else:
            for each in data:
                f.write(each + '\n')


def get_data_by_type():
    raw_train = './train_seg.csv'
    raw_test = './test_seg.csv'

    sp = []
    ly = []
    jr = []
    yl = []
    wl = []

    with open(raw_train, 'r') as f:
        for line in f:
            idx, typ, comment, label = line.split('\t')
            if typ == '食品餐饮':
                sp.append(comment + '\t__label__' + label)
            elif typ == '旅游住宿':
                ly.append(comment + '\t__label__' + label)
            elif typ == '金融服务':
                jr.append(comment + '\t__label__' + label)
            elif typ == '医疗服务':
                yl.append(comment + '\t__label__' + label)
            elif typ == '物流快递':
                wl.append(comment + '\t__label__' + label)
            else:
                print('unnormal data: %s' % line)

    for i, ls in enumerate([sp, ly, jr, yl, wl]):
        with open('./train_type_%d' % i, 'a') as f:
            for each in ls:
                f.write(each)
    total = len(sp) + len(ly) + len(jr) + len(yl) + len(wl)
    print('train data: %d' % total)

    sp = []
    ly = []
    jr = []
    yl = []
    wl = []

    with open(raw_test, 'r') as f:
        for line in f:
            idx, typ, comment = line.split('\t')
            if typ == '食品餐饮':
                sp.append(comment)
            elif typ == '旅游住宿':
                ly.append(comment)
            elif typ == '金融服务':
                jr.append(comment)
            elif typ == '医疗服务':
                yl.append(comment)
            elif typ == '物流快递':
                wl.append(comment)
            else:
                print('unnormal data: %s' % line)
    total = len(sp) + len(ly) + len(jr) + len(yl) + len(wl)
    print('test data: %d' % total)
    for i, ls in enumerate([sp, ly, jr, yl, wl]):
        with open('./test_type_%d' % i, 'a') as f:
            for each in ls:
                f.write(each)


# get_data_by_type()
def divide(path, tpath, vpath, rate=0.8):
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            data.append(line)
    n = len(data)
    print(n)
    ntrain = int(n * rate)
    print('训练集数据量： %d' % ntrain)
    random.shuffle(data)

    with open(tpath, 'w') as f:
        for line in data[:ntrain]:
            f.write(line)
    with open(vpath, 'w') as f:
        for line in data[ntrain:]:
            f.write(line)


