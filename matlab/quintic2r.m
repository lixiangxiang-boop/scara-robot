function [q, dq, ddq] = quintic2r(q0, q1, T, t)
%QUINTIC2R 关节空间五次多项式（首末速度/加速度为零）
%   q0,q1: n×1 关节角；t: 任意形状时间序列
%   返回 q,dq,ddq: n×N（N = numel(t)）
tau = reshape(min(max(t(:).'/T, 0), 1), 1, []);      % 1×N 行向量
h = q1(:) - q0(:);                                   % n×1 列向量
s   = 10*tau.^3 - 15*tau.^4 + 6*tau.^5;
ds  = (30*tau.^2 - 60*tau.^3 + 30*tau.^4) / T;
dds = (60*tau - 180*tau.^2 + 120*tau.^3) / T^2;
q = q0(:) + h*s;  dq = h*ds;  ddq = h*dds;
end
