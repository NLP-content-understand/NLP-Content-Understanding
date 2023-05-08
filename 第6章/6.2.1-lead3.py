#引入切分句子的代码
from nlg_yongzhuo.data_preprocess.text_preprocess import cut_sentence
#Lead3提取文本摘要
class Lead3Sum:
    def __init__(self):
        self.algorithm = 'lead_3'

    def summarize(self, text, type='mix', num=3):
        """
            lead-s
        :param sentences: list
        :param type: str, you can choose 'begin', 'end' or 'mix'
        :return: list
        """
        sentences = cut_sentence(text)
        if len(sentences) < num:
            return sentences
        # 最小句子数
        num_min = min(num, len(sentences))
        if type == 'begin':
            summers = sentences[0:num]
        elif type == 'end':
            summers = sentences[-num:]
        else:
            summers = [sentences[0]] + [sentences[-1]] + sentences[1:num-1]
        summers_s = {}
        for i in range(len(summers)): # 得分计算
            if len(summers) - i == 1:
                summers_s[summers[i]] = (num - 0.75) / (num + 1)
            else:
                summers_s[summers[i]] = (num - i - 0.5) / (num + 1)
        score_sen = [(rc[1], rc[0]) for rc in sorted(summers_s.items(), key=lambda d: d[1], reverse=True)][0:num_min]
        return score_sen

if __name__ == '__main__':
    doc = "是上世纪90年代末提出的一种计算网页权重的算法。" \
          "当时，互联网技术突飞猛进，各种网页网站爆炸式增长，" \
          "业界急需一种相对比较准确的网页重要性计算方法，" \
          "是人们能够从海量互联网世界中找出自己需要的信息。" \
          "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。" \
          "Google把从A页面到B页面的链接解释为A页面给B页面投票，" \
          "Google根据投票来源甚至来源的来源，即链接到A页面的页面" \
          "和投票目标的等级来决定新的等级。简单的说，" \
          "一个高等级的页面可以使其他低等级页面的等级提升。" \
          "PageRank The PageRank Citation Ranking: Bringing Order to the Web，"\
          "具体说来就是，PageRank有两个基本思想，也可以说是假设，" \
          "即数量假设：一个网页被越多的其他页面链接，就越重）；" \
          "质量假设：一个网页越是被高质量的网页链接，就越重要。" \
          "总的来说就是一句话，从全局角度考虑，获取重要的信息。"
    doc = doc.encode('utf-8').decode('utf-8')
    l3 = Lead3Sum()
    for score_sen in l3.summarize(doc, type='mix', num=3):
        print(score_sen)
