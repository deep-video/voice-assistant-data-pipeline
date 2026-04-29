# 所有意图的描述配置
INTENT_DESCRIPTIONS = {
    "music_recommendation": {
        "description": "用户想要系统推荐/建议音乐，而非直接播放。包含但不限于：询问听什么、推荐几首、旋律、歌、音乐等表达",
        "intent_action": "系统给出音乐推荐/建议列表",
        "intent_negative_example": "直接播放音乐（如'来首歌'、'放点音乐'、'播放XX'）"
    },
    "travel_planning": {
        "description": "用户想要系统规划旅行路线或行程",
        "intent_action": "系统提供旅行规划方案",
        "intent_negative_example": "直接查询交通信息（如'怎么去机场'、'现在路况如何'）"
    },
    "poi_recommendation": {
        "description": "用户想要系统推荐/建议兴趣点（POI）",
        "intent_action": "系统给出POI推荐列表",
        "intent_negative_example": "直接查询POI信息（如'附近有什么餐厅'、'XX在哪里'）"
    },
    "history_query": {
        "description": "用户想要查询历史记录（如播放历史、搜索历史）",
        "intent_action": "系统提供历史记录信息",
        "intent_negative_example": "直接执行操作（如'继续播放上次的歌'、'回到上一页'）"
    },
    "weather_query": {
        "description": "用户想要查询天气信息",
        "intent_action": "系统提供天气信息",
        "intent_negative_example": "直接执行操作（如'打开空调'、'关闭窗户'）"
    },
    "traffic_query": {
        "description": "用户想要查询交通信息（如路况、公交、地铁）",
        "intent_action": "系统提供交通信息",
        "intent_negative_example": "直接执行操作（如'导航到XX'、'避开拥堵'）"
    },
    "car_control": {
        "description": "用户想要控制车辆功能（如空调、车窗、座椅）",
        "intent_action": "系统执行车辆控制操作",
        "intent_negative_example": "直接查询车辆信息（如'空调温度是多少'、'车窗是否关闭'、'播放XX'）"
    },
    "phone_call": {
        "description": "用户想要拨打电话",
        "intent_action": "系统执行拨打电话操作",
        "intent_negative_example": "直接查询电话信息（如'XX的电话是多少'、'最近的通话记录'）"
    },
    "emergency_assistance": {
        "description": "用户想要寻求紧急帮助",
        "intent_action": "系统提供紧急帮助",
        "intent_negative_example": "直接查询紧急信息（如'最近的医院在哪里'、'急救电话是多少'）"
    },
    "transport_query": {
        "description": "用户想要查询交通方式信息（如飞机、火车、公交）",
        "intent_action": "系统提供交通方式信息",
        "intent_negative_example": "直接执行操作（如'预订机票'、'购买火车票'）"
    },
    "media_recommendation": {
        "description": "用户想要系统推荐/建议媒体内容（如视频、播客，不包括音乐）",
        "intent_action": "系统给出媒体推荐/建议列表",
        "intent_negative_example": "直接播放媒体（如'播放XX'、'来点视频'）"
    }
}

# 所有意图列表
ALL_INTENTS = list(INTENT_DESCRIPTIONS.keys())