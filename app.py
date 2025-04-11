import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from predict import ImagePredictor
from utils.config import MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__, static_folder='static')

# 确保上传目录存在
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 添加对pre_data目录的静态文件访问支持
from flask import send_from_directory

@app.route('/pre_data/<path:filename>')
def serve_pre_data(filename):
    return send_from_directory('pre_data', filename)

# 地域和季节配置
REGIONS = ['华北', '东北', '华东', '华中', '华南', '西南', '西北']
SEASONS = ['春季', '夏季', '秋季', '冬季']

# 舌象分类映射
TONGUE_CLASSES = {
    0: '淡白舌灰黑苔',
    1: '淡白舌白苔',
    2: '淡白舌黄苔',
    3: '淡红舌灰黑苔',
    4: '淡红舌白苔',
    5: '淡红舌黄苔',
    6: '红舌灰黑苔',
    7: '红舌白苔',
    8: '红舌黄苔',
    9: '绛舌灰黑苔',
    10: '绛舌白苔',
    11: '绛舌黄苔',
    12: '青紫舌灰黑苔',
    13: '青紫舌白苔',
    14: '青紫舌黄苔'
}

# 全局变量
device = None
model = None

# 初始化图像预测器
def init_predictor():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pth')
        predictor = ImagePredictor(model_path)
        logging.info('图像预测器初始化成功')
        return predictor
    except Exception as e:
        logging.error(f'图像预测器初始化失败：{str(e)}')
        return None

@app.route('/')
def index():
    # 获取当前季节
    current_month = datetime.now().month
    current_season = (current_month - 1) // 3
    return render_template('index.html', regions=REGIONS, seasons=SEASONS, current_season=current_season)

@app.route('/get_test_categories')
def get_test_categories():
    try:
        test_dir = os.path.join('pre_data', 'test')
        if not os.path.exists(test_dir):
            return jsonify({'error': '测试集目录不存在'}), 404
        
        categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_test_images/<category>')
def get_test_images(category):
    try:
        category_dir = os.path.join('pre_data', 'test', category)
        if not os.path.exists(category_dir):
            return jsonify({'error': '类别目录不存在'}), 404
        
        images = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_urls = [f'/pre_data/test/{category}/{img}' for img in images]
        # 确保图片存在
        for img in images:
            if not os.path.exists(os.path.join(category_dir, img)):
                return jsonify({'error': f'图片文件不存在：{img}'}), 404
        return jsonify({'images': image_urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 在应用启动时初始化预测器
predictor = init_predictor()
if predictor is None:
    print('预测器初始化失败，应用将在收到请求时尝试重新初始化')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'tongue_data' not in request.files:
        return jsonify({'error': '请上传舌象图片'}), 400
    
    file = request.files['tongue_data']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 检查预测器是否正确初始化
    global predictor
    if predictor is None:
        return jsonify({'error': '系统未准备就绪，请稍后重试'}), 503
    
    try:
        # 获取地域和季节信息
        region_index = int(request.form.get('region', 0))
        season_index = int(request.form.get('season', 0))
        
        # 保存上传的文件
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        try:
            # 使用预测器进行预测
            prediction = predictor.predict_single(filename)
            constitution_type = prediction['class']
            probability = prediction['probability']
            
            # 构建分析结果
            features = get_constitution_features(constitution_type)
            top_constitutions = [{
                'type': constitution_type,
                'probability': probability,
                'features': features
            }]
            
            # 生成养生建议
            advice = generate_health_advice(
                constitution_type,
                REGIONS[region_index],
                SEASONS[season_index]
            )
            
            # 渲染结果页面
            return render_template('result.html',
                                 result={
                                     'image_url': f'/static/uploads/{file.filename}',
                                     'top_constitutions': top_constitutions,
                                     'advice': advice
                                 },
                                 region=REGIONS[region_index],
                                 season=SEASONS[season_index])
                                 
        except Exception as e:
            return jsonify({'error': f'预测失败：{str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': f'服务器处理错误：{str(e)}'}), 500
    finally:
        # 清理临时文件
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception as e:
            print(f'清理临时文件失败：{str(e)}')

def get_constitution_features(constitution_type):
    """根据体质类型返回对应的舌象特征描述"""
    features = {
        '淡白舌灰黑苔': '舌质淡白，苔色灰黑，提示气血两虚，寒凝血瘀',
        '淡白舌白苔': '舌质淡白，苔色洁白，提示气血亏虚，脾胃虚寒',
        '淡白舌黄苔': '舌质淡白，苔色黄腻，提示气血不足，湿热内蕴',
        '淡红舌灰黑苔': '舌质淡红，苔色灰黑，提示正气不足，瘀血内停',
        '淡红舌白苔': '舌质淡红，苔色白润，提示气血调和，脾胃功能正常',
        '淡红舌黄苔': '舌质淡红，苔色黄腻，提示湿热内蕴，脾胃运化不足',
        '红舌灰黑苔': '舌质红赤，苔色灰黑，提示热毒炽盛，血瘀内停',
        '红舌白苔': '舌质红赤，苔色白腻，提示阳热偏盛，湿邪内蕴',
        '红舌黄苔': '舌质红赤，苔色黄厚，提示热毒炽盛，湿热内蕴',
        '绛舌灰黑苔': '舌质绛紫，苔色灰黑，提示热毒炽盛，血瘀内停',
        '绛舌白苔': '舌质绛紫，苔色白腻，提示阴虚火旺，湿邪内蕴',
        '绛舌黄苔': '舌质绛紫，苔色黄厚，提示阴虚火旺，湿热内蕴',
        '青紫舌灰黑苔': '舌质青紫，苔色灰黑，提示寒凝血瘀，气血瘀滞',
        '青紫舌白苔': '舌质青紫，苔色白腻，提示寒凝血瘀，痰湿内停',
        '青紫舌黄苔': '舌质青紫，苔色黄腻，提示血瘀痰浊，湿热内蕴'
    }
    return features.get(constitution_type, '暂无特征描述')

def generate_health_advice(constitution_type, region, season):
    """根据体质类型、地域和季节生成个性化养生建议"""
    # 基础建议模板
    base_advice = {
        '饮食调养': {
            '淡白舌': '宜食用温补气血的食物，如红枣、桂圆、羊肉等',
            '淡红舌': '饮食宜清淡，可适当食用滋补气血的食物',
            '红舌': '忌食辛辣刺激，宜食用清淡凉性食物',
            '绛舌': '宜食用滋阴降火的食物，如梨、银耳、莲子等',
            '青紫舌': '宜食用活血化瘀的食物，如红枣、当归、桃仁等'
        },
        '起居调养': {
            '春季': '早睡早起，适当运动，注意保暖',
            '夏季': '晚睡早起，注意防暑降温，适量运动',
            '秋季': '早睡早起，注意保暖，适当运动',
            '冬季': '早睡晚起，注意保暖，室内适当运动'
        },
        '地域特点': {
            '华北': '气候干燥，注意保湿，冬季特别注意保暖',
            '东北': '气候寒冷，注意保暖，室内要保持适宜温度',
            '华东': '气候湿润，注意防潮，梅雨季节特别注意',
            '华中': '四季分明，注意适应季节变化',
            '华南': '气候炎热，注意防暑降温，注意补充水分',
            '西南': '气候潮湿，注意防潮，保持室内通风',
            '西北': '气候干燥，注意保湿，防风沙'
        }
    }
    
    # 获取体质基本类型（去除苔色信息）
    base_type = constitution_type[:3]
    
    # 组合建议
    advice = f"""1. 饮食调养：
{base_advice['饮食调养'].get(base_type, '保持均衡饮食，清淡为主')}

2. 起居作息：
{base_advice['起居调养'][season]}

3. 地域特点：
{base_advice['地域特点'][region]}

4. 注意事项：
- 保持良好的作息规律
- 适度运动，避免过度劳累
- 保持心情舒畅，避免情绪波动
- 定期进行舌象检查，关注身体变化"""
    
    return advice

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5000)