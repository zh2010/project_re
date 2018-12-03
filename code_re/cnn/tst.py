# -*- coding: utf-8 -*-
import re

if __name__ == '__main__':
    s1 = 'GnRH脉冲输注技术'
    a = re.search('[^\u4e00-\u9fa5]+', s1)
    s2 = '11β-羟化酶缺陷症'
    b = re.search('[^\u4e00-\u9fa5]+', s2)
    s3 = 'm-FG'
    c = re.search('[^\u4e00-\u9fa5]+', s3)
    print('done')
