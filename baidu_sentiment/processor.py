import helper

train_path = '/home/ytj/data/data_train.csv'
test_path = '/home/ytj/data/data_test.csv'

ctrain_path = './train_clean.csv'
ctest_path = './test_clean.csv'

seg_train_path = './train_seg.csv'
seg_test_path = './test_seg.csv'

train_fasttext = './train_ft.csv'
test_fasttext = './test_ft.csv'

helper.clean_train_data(train_path, ctrain_path)
helper.clean_test_data(test_path, ctest_path)

stop_words = [',', '\.', '，', '。', 'hellip', '&', '&hellip', '？', '！', '?', '!', '……', '__&', '(', ')', '（', '）', '⊙',
              'o', '~', '~', '\"', '“', '”', '、', 'bull', '_&', 'oline', '了', '的', ' ', '_', '...', '\\', '/', '__',
              '呀', '啊', '-', 'ldquo', 'rdquo', ':', '：', ';', '；', '【', '】', 'middot', '`', 'nbsp', '哦']
helper.seg(ctrain_path, seg_train_path, True, stop_words)
helper.seg(ctest_path, seg_test_path, False, stop_words)

helper.fit_fasttext(seg_train_path, train_fasttext, isTraining=True)
helper.fit_fasttext(seg_test_path, test_fasttext, isTraining=False)
