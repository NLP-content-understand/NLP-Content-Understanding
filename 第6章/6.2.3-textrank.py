import re
import networkx

text = '虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。那么，第一款夏普手机什么时候登陆中国呢？又会是怎么样的手机呢？近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，采用 14 纳米工艺，八个 Kryo 260 核心设计，集成 Adreno 512 GPU 和 X12 LTE 调制解调器。当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，可以独占两三个月时间。考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 年推出全球首款全面屏手机 EDGEST 302SH 至今，夏普手机推出了多达 28 款的全面屏手机。在 5 月份的媒体沟通会上，惠普罗忠生表示：“我敢打赌，12 个月之后，在座的各位手机都会换掉。因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。”'
def split_sentences(text, p='[。.，,？：]', filter_p='\s+'):
    f_p = re.compile(filter_p)
    text = re.sub(f_p, '', text)
    pattern = re.compile(p)
    split = re.split(pattern, text)
    return split

def get_sen_graph(text,window=3):
    split_sen = split_sentences(text)
    sentences_graph = networkx.graph.Graph()
    for i,sen in enumerate(split_sen):
        sentences_graph.add_edges_from([(sen,split_sen[ii])
        for ii in range(i-window,i+window+1)
            if ii >= 0 and ii < len(split_sen)])
    return sentences_graph

def text_rank(text):
    sentences_graph = get_sen_graph(text)
    ranking_sentences = networkx.pagerank(sentences_graph)
    ranking_sentences_sorted = sorted(ranking_sentences.items(), key=lambda x:x[1], reverse=True)
    return ranking_sentences_sorted

def get_summarization(text,score_fn,sum_len):
    sub_sentences = split_sentences(text)
    ranking_sentences = score_fn(text)
    selected_sen = set()
    current_sen = ''
    for sen, _ in ranking_sentences:
        if len(current_sen)<sum_len:
            current_sen += sen
            selected_sen.add(sen)
        else:
            break
        summarized = []
        for sen in sub_sentences:
            if sen in selected_sen:
                summarized.append(sen)
    return summarized
def get_summarization_by_text_rank(text,sum_len=200):

    return get_summarization(text,text_rank,sum_len)

print(' '.join(get_summarization_by_text_rank(text)))
