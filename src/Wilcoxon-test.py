from scipy.stats import wilcoxon

# Métricas de desempenho (sem desvios padrão)
acc_artigos = [0.616, 0.741, 0.735, 0.732, 0.768]
bacc_artigos = [0.651, 0.728, 0.765, 0.742, 0.775]
auc_artigos = [0.901, 0.929, 0.935, 0.936, 0.947]

# Suas métricas
acc_suas = [0.7977] * len(acc_artigos)  # Repetir o valor do seu modelo
bacc_suas = [0.7837] * len(bacc_artigos)  # Repetir o valor do seu modelo
auc_suas = [0.9387] * len(auc_artigos)  # Repetir o valor do seu modelo

# Realizando o teste de Wilcoxon para ACC, BACC e AUC
stat_acc, p_acc = wilcoxon(acc_artigos, acc_suas)
stat_bacc, p_bacc = wilcoxon(bacc_artigos, bacc_suas)
stat_auc, p_auc = wilcoxon(auc_artigos, auc_suas)

# Resultados
print(f"Estatística ACC: {stat_acc}, Valor de p: {p_acc}")
print(f"Estatística BACC: {stat_bacc}, Valor de p: {p_bacc}")
print(f"Estatística AUC: {stat_auc}, Valor de p: {p_auc}")

# Interpretação
if p_acc < 0.05:
    print("Diferença significativa em ACC.")
else:
    print("Diferença não significativa em ACC.")

if p_bacc < 0.05:
    print("Diferença significativa em BACC.")
else:
    print("Diferença não significativa em BACC.")

if p_auc < 0.05:
    print("Diferença significativa em AUC.")
else:
    print("Diferença não significativa em AUC.")
