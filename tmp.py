from graphviz import Digraph

dot = Digraph(comment='联邦学习流程图')

dot.node('A', 'C 客户端1')
dot.node('B', 'C 中央服务器')
dot.node('C', 'C 客户端2')
dot.node('D', 'C 客户端3')

dot.edges(['AB', 'BA', 'BC', 'CB', 'BD', 'DB'])

dot.edge('A', 'B', label='2. 利用本地数据训练模型\n3. 发送新的模型\n5. 分发更新后的全局模型')
dot.edge('B', 'A', label='1. 初始化全局模型并分发给客户端\n4. 聚合并来自客户端的模型，以更新全局模型\n5. 分发更新后的全局模型')
dot.edge('B', 'C', label='5. 分发更新后的全局模型\n3. 发送新的模型\n5. 分发更新后的全局模型')
dot.edge('C', 'B', label='2. 利用本地数据训练模型\n3. 发送新的模型\n5. 分发更新后的全局模型')
dot.edge('B', 'D', label='5. 分发更新后的全局模型\n3. 发送新的模型\n5. 分发更新后的全局模型')
dot.edge('D', 'B', label='2. 利用本地数据训练模型\n3. 发送新的模型\n5. 分发更新后的全局模型')

dot.render('federated_learning', view=True)
