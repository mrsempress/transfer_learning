def f_train(X_src, y_src, X_tgt0, X_tgt1):
    X_src = torch.tensor(X_src, dtype=torch.float, device=torch_device)
    y_src = torch.tensor(y_src, dtype=torch.long, device=torch_device)
    X_tgt0 = torch.tensor(X_tgt0, dtype=torch.float, device=torch_device)
    X_tgt1 = torch.tensor(X_tgt1, dtype=torch.float, device=torch_device)

    student_optimizer.zero_grad()
    student_net.train()
    teacher_net.train()

    src_logits_out = student_net(X_src)
    student_tgt_logits_out = student_net(X_tgt0)
    student_tgt_prob_out = F.softmax(student_tgt_logits_out, dim=1)
    teacher_tgt_logits_out = teacher_net(X_tgt1)
    teacher_tgt_prob_out = F.softmax(teacher_tgt_logits_out, dim=1)

    # Supervised classification loss
    if double_softmax:
        clf_loss = classification_criterion(F.softmax(src_logits_out, dim=1), y_src)
    else:
        clf_loss = classification_criterion(src_logits_out, y_src)

    unsup_loss, conf_mask_count, unsup_mask_count = compute_aug_loss(student_tgt_prob_out, teacher_tgt_prob_out)

    loss_expr = clf_loss + unsup_loss * unsup_weight

    loss_expr.backward()
    student_optimizer.step()
    teacher_optimizer.step()

    n_samples = X_src.size()[0]

    outputs = [float(clf_loss) * n_samples, float(unsup_loss) * n_samples]
    if not use_rampup:
        mask_count = float(conf_mask_count)
        unsup_count = float(unsup_mask_count)

        outputs.append(mask_count)
        outputs.append(unsup_count)
    return tuple(outputs)
