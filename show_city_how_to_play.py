province_city_dict = {
    '北京': ['北京'],
    '天津': ['天津'],
    '河北': ['石家庄', '唐山', '秦皇岛', '邯郸', '邢台', '保定', '张家口', '承德', '沧州', '廊坊', '衡水'],
    '山西': ['太原', '大同', '阳泉', '长治', '晋城', '朔州', '晋中', '运城', '忻州', '临汾', '吕梁'],
    '内蒙古': ['呼和浩特', '包头', '乌海', '赤峰', '通辽', '鄂尔多斯', '呼伦贝尔', '巴彦淖尔', '乌兰察布', '兴安盟', '锡林郭勒盟', '阿拉善盟'],
    '辽宁': ['沈阳', '大连', '鞍山', '抚顺', '本溪', '丹东', '锦州', '营口', '阜新', '辽阳', '盘锦', '铁岭', '朝阳', '葫芦岛'],
    '吉林': ['长春', '吉林', '四平', '辽源', '通化', '白山', '松原', '白城', '延边朝鲜族自治州'],
    '黑龙江': ['哈尔滨', '齐齐哈尔', '鸡西', '鹤岗', '双鸭山', '大庆', '伊春', '佳木斯', '七台河', '牡丹江', '黑河', '绥化', '大兴安岭'],
    '上海': ['上海'],
    '江苏': ['南京', '无锡', '徐州', '常州', '苏州', '南通', '连云港', '淮安', '盐城', '扬州', '镇江', '泰州', '宿迁'],
    '浙江': ['杭州', '宁波', '温州', '嘉兴', '湖州', '绍兴', '金华', '衢州', '舟山', '台州', '丽水'],
    '安徽': ['合肥', '芜湖', '蚌埠', '淮南', '马鞍山', '淮北', '铜陵', '安庆', '黄山', '滁州', '阜阳', '宿州', '巢湖', '六安', '亳州', '池州', '宣城'],
    '福建': ['福州','漳州', '南平', '龙岩', '宁德'],
'江西': ['南昌', '景德镇', '萍乡', '九江', '新余', '鹰潭', '赣州', '吉安', '宜春', '抚州', '上饶'],
'山东': ['济南', '青岛', '淄博', '枣庄', '东营', '烟台', '潍坊', '济宁', '泰安', '威海', '日照', '莱芜', '临沂', '德州', '聊城', '滨州', '菏泽'],
'河南': ['郑州', '开封', '洛阳', '平顶山', '安阳', '鹤壁', '新乡', '焦作', '濮阳', '许昌', '漯河', '三门峡', '南阳', '商丘', '信阳', '周口', '驻马店'],
'湖北': ['武汉', '黄石', '十堰', '宜昌', '襄阳', '鄂州', '荆门', '孝感', '荆州', '黄冈', '咸宁', '随州', '恩施土家族苗族自治州', '仙桃', '潜江', '天门', '神农架林区'],
'湖南': ['长沙', '株洲', '湘潭', '衡阳', '邵阳', '岳阳', '常德', '张家界', '益阳', '郴州', '永州', '怀化', '娄底', '湘西土家族苗族自治州'],
'广东': ['广州', '韶关', '深圳', '珠海', '汕头', '韩城', '佛山', '江门', '湛江', '茂名', '肇庆', '惠州', '梅州', '汕尾', '河源', '阳江', '清远', '东莞', '中山', '潮州', '揭阳', '云浮'],
'广西': ['南宁', '柳州', '桂林', '梧州', '北海', '防城港', '钦州', '贵港', '玉林', '百色', '贺州', '河池', '来宾', '崇左'],
'海南': ['海口', '三亚', '三沙', '儋州', '五指山', '琼海', '文昌', '万宁', '东方', '定安县', '屯昌县', '澄迈县', '临高'],
'西藏': ['拉萨', '日喀则', '昌都', '林芝', '山南', '那曲', '阿里'],
'陕西': ['西安', '铜川', '宝鸡', '咸阳', '渭南', '延安', '汉中', '榆林', '安康', '商洛'],
'甘肃': ['兰州', '嘉峪关', '金昌', '白银', '天水', '武威', '张掖', '平凉', '酒泉', '庆阳', '定西', '陇南', '临夏回族自治州', '甘南藏族自治州'],
'青海': ['西宁', '海东', '海北藏族自治州', '黄南藏族自治州', '海南藏族自治州', '果洛藏族自治州', '玉树藏族自治州', '海西蒙古族藏族自治州'],
'宁夏': ['银川', '石嘴山', '吴忠', '固原', '中卫'],
'新疆': ['乌鲁木齐', '克拉玛依', '吐鲁番', '哈密', '昌吉回族自治州', '博尔塔拉蒙古自治州', '巴音郭楞蒙古自治州', '阿克苏地区', '克孜勒苏柯尔克孜自治州', '喀什地区', '和田地区', '伊犁哈萨克自治州', '塔城地区', '阿勒泰地区', '自治区直辖县级行政区划']
}

import os
import platform
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

os_name = platform.system()
print("os: " + os_name)
previous_time = datetime.now()

print(f"begin load. {datetime.now()}")
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", cache_dir="G:/GPT/THUDM_chatglm-6b", trust_remote_code=True)
print(f"\nload tokenizer ok. cose:{ (datetime.now() - previous_time).total_seconds()} sec\n")

previous_time = datetime.now()

model = AutoModel.from_pretrained("THUDM/chatglm-6b",  
                                  cache_dir="G:/GPT/THUDM_chatglm-6b", 
                                  trust_remote_code=True
                                  ).half().quantize(8).cuda()
# 进行 2 至 3 轮对话后，8-bit 量化下 GPU 显存占用约为 10GB，4-bit 量化下仅需 6GB 占用。随着对话轮数的增多，对应消耗显存也随之增长，由于采用了相对位置编码，理论上 ChatGLM-6B 支持无限长的 context-length，但总长度超过 2048（训练长度）后性能会逐渐下降。
print(f"\nload model ok: cost: { (datetime.now() - previous_time).total_seconds() } sec\n")

model = model.eval()

file = open('qa.txt', 'a')
def writefile(city, context, sec):
    file.write(f" \n({city}) cost:{sec} sec\n {context} \n \n")
    file.flush()
    
for province, cities in province_city_dict.items():
    for city in cities:
        history = []
        
        previous_time = datetime.now()
        query = f"我要去{province}的{city}旅游，有什么推荐的地方？"
        if province == city:
            query = f"我要去{province}旅游，有什么推荐的地点？"
        
        response, history = model.chat(tokenizer, query, history=history)
        sec = (datetime.now() - previous_time).total_seconds()
        print(f"ChatGLM-6B：{response} \n cost:{ sec } sec\n")
        
        if province == city:
            # 直辖市会多问一句
            query = '还有其它的推荐地方吗？'     
            previous_time = datetime.now()       
            response2, history = model.chat(tokenizer, query, history=history)
            sec += (datetime.now() - previous_time).total_seconds()
            print(f"ChatGLM-6B：{response2} \n cost:{ (datetime.now() - previous_time).total_seconds() } sec\n")
            writefile(city, response + response2, sec)
        else:
            writefile(city, response, sec)
        
file.close()