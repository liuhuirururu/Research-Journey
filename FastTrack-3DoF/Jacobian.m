function J = Jacobian(q1, q2, q3, L1, L2, L3)
% 计算平面三连杆机器人的雅可比矩阵
% q: [q1; q2; q3] 关节角度 
% L: [L1; L2; L3] 连杆长度



% 雅可比矩阵计算公式
J11 = -L1*sin(q1)-L2*sin(q1+q2)-L3*sin(q1+q2+q3);
J12 = -L2*sin(q1+q2)-L3*sin(q1+q2+q3);
J13 = -L3*sin(q1+q2+q3);
 
J21 = L1*cos(q1)+L2*cos(q1+q2)+L3*cos(q1+q2+q3);
J22 = L2*cos(q1+q2)+L3*cos(q1+q2+q3);
J23 = L3*cos(q1+q2+q3);

J = [J11 J12 J13;
    J21 J22 J23];

end