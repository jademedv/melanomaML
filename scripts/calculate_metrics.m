% response: true labels (benign/malignant)
% prediction: predicted labels
% score: malignant class score for AUC
% metrics: table with results
function metrics = calculate_metrics(response, prediction, score)

class = categorical({'benign'; 'malignant'});
% confusion matrix
C = confusionmat(response, prediction, 'Order', class);
tn = C(1,1);
fp = C(1,2);
fn = C(2,1);
tp = C(2,2);
% metrics from confusion matrix
accuracy = (tp + tn) / sum(C, 'all');
[recall, ci] = binofit(tp, tp + fn);
recall_lo = ci(1);
recall_hi = ci(2);
precision = tp / (tp + fp);
specificity = tn / (tn + fp);
f1 = 2 * precision * recall / (precision + recall);
% AUC is only calculated when a score is supplied
auc = NaN;
if nargin > 2
    [~, ~, ~, auc] = perfcurve(response, score, 'malignant');
end
%return the metrics as a table
metrics = table(accuracy, recall, recall_lo, recall_hi, precision, specificity, f1, auc, ...
    'VariableNames', {'Accuracy', 'Recall', 'RecallLo', 'RecallHi', ...
    'Precision', 'Specificity', 'F1', 'AUC'});
end
