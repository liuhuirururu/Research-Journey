function dqc = MyAnswer(q, dq, target_pos, reach_status, link_length, sample_time)
%   STUDENTANSWER Summary of this function goes here
%   Detailed explanation goes here

%%  Input Explanation
%   q - （当前关节角）Current robot joint value. 3 * 1 vector 
%   dq -（当前关节速度） Current robot joint velocity. 3 * 1 vector
%   target_pos - （目标点位置）The target position for this problem. 2 * 15 matrix. Each
%   columnum is one target position. 
%   reach_status -（每个目标位置的访问状态） The visited status of each target position. 
%   0 - unvisited, 1 visited.
%   link_length - （关节长度）The link length of robot
%   sample_time - （取样时间）The sample time of current control loop.
%
%   dqc -（输出-关节速度） The only output variable, which stands for the velocity of robot
%   joint. 3 * 1 vector.


%%  Write your answer here.

%% ----------------Step.1 计算当前末端位置-------------------
theta1 = q(1); 
theta2 = q(2); 
theta3 = q(3); %取角度
L1 = link_length(1); 
L2 = link_length(2); 
L3 = link_length(3); %取连杆长
% 平面三连杆机器人正运动学计算
x = L1*cos(theta1) + L2*cos(theta1+theta2) + L3*cos(theta1+theta2+theta3);
y = L1*sin(theta1) + L2*sin(theta1+theta2) + L3*sin(theta1+theta2+theta3);
current_pos = [x; y];

%% ----------------Step.2 筛选未访问目标点-------------------
unvisited_idx = find(reach_status == 0);
if isempty(unvisited_idx)
    dqc = zeros(3,1);  % 全部到达则关节速度为0
    return;
end
unvisited_targets = target_pos(:, unvisited_idx);

%% ----------------Step.3 计算末端误差向量-------------------
num_targets = length(unvisited_idx);
error_vectors = zeros(2, num_targets);
for k = 1:num_targets
    error_vectors(:, k) = unvisited_targets(:, k) - current_pos;
end

%% ----------------Step.4 计算雅可比矩阵-------------------
J = Jacobian(theta1, theta2, theta3, L1, L2, L3);  

%% ----------------Step.5 计算损失并选择目标-------------------
alpha = 0.39;        % 权重系数
threshold = sum(link_length) * 0.96;  % 定义阈值大小（以关节总长度为基础）舍弃过远目标

cost = inf(1, num_targets);  % 初始化代价数组
dq_candidates = zeros(3, num_targets);

for k = 1:num_targets
    % 判断距离阈值
    %unvisited_targets
    if norm(unvisited_targets(:, k)) > threshold
        continue;   % 超过阈值舍弃
    end
    
    % 计算末端期望速度
    v_k = error_vectors(:, k);    
    dq_k = pinv(J) * v_k;            
     
    dq_candidates(:, k) = dq_k;         % 保存候选关节速度
    
    % cost = 关节速度二范数 + alpha * 末端几何距离
    cost(k) = norm(dq_k) + alpha * norm(v_k);
end

%% ----------------Step.6 选取最小代价目标-------------------
[~, best_idx_local] = min(cost);              % 找到代价最小的目标

if isinf(cost(best_idx_local))
    dqc = zeros(3,1);  % 如果所有点都舍弃，关节速度为0
    return;
end

best_dq = dq_candidates(:, best_idx_local);         % 取出对应的关节速度
dqc = 120* best_dq;

end

