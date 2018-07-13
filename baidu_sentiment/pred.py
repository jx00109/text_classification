import fasttext
import helper

MODEL = 'all'
N = 10
if MODEL == 'all':
    for i in range(N):
        train_data_path = './files/train_ft.csv'
        train_path = './files/train_8_%d.csv' % i
        valid_path = './files/valid_2_%d.csv' % i
        test_path = './files/test_ft.csv'
        helper.divide(train_data_path, train_path, valid_path)

        classifier = fasttext.supervised(train_path, "./model/ft_%s_%d.model" % (MODEL, i + 1),
                                         label_prefix="__label__", loss='hs', lr=0.01, dim=300, epoch=300, word_ngrams=3,
                                         bucket=100000, ws=7)
        valid_acc = classifier.test(valid_path).precision
        data = []
        with open(test_path, 'r') as f:
            for line in f:
                data.append(line)

        preds = [e[0] for e in classifier.predict(data)]
        count = 1
        with open('./results/results_%s_%.4f.csv' % (MODEL, valid_acc), 'w') as f:
            for each in preds:
                for p in each:
                    f.write(str(count) + "," + p + '\n')
                    count += 1
else:
    preds = []
    for i in range(5):
        classifier = fasttext.supervised('./files/train_type_%d' % i, "./model/ft_type_%d.model" % i,
                                         label_prefix="__label__",loss='hs', lr=0.1, dim=300, epoch=300, word_ngrams=3,
                                         bucket=100000)
        data = []
        with open('./files/test_type_%d' % i, 'r') as f:
            for line in f:
                data.append(line)

        pred = [e[0] for e in classifier.predict(data)]
        preds.append(pred)
    count = 1
    with open('./results/results_%s.csv' % MODEL, 'a') as f:
        for each in preds:
            for p in each:
                f.write(str(count) + "," + p + '\n')
                count += 1
