import numpy as np


def mask_retrieval(class_response_maps, peak_list, peak_response_maps, retrieval_cfg):
        # cast tensors to numpy array
        class_response_maps = class_response_maps.squeeze().cpu().numpy()
        peak_list = peak_list.cpu().numpy()
        peak_response_maps = peak_response_maps.cpu().numpy()

        img_height, img_width = peak_response_maps.shape[1], peak_response_maps.shape[2]
        img_area = img_height * img_width

        # parse configuration
        proposals = retrieval_cfg['proposals']  # segment proposals off-the-shelf
        contour_width = retrieval_cfg.get('contour_width', 5)  # proposal contour width
        proposal_size_limit = retrieval_cfg.get('proposal_size_limit', (0.00002, 0.85))  # limit range of proposal size
        proposal_count = retrieval_cfg.get('proposal_count', 100)  # selected number of proposals
        nms_threshold = retrieval_cfg.get('nms_threshold', 0.3)  # nms threshold
        merge_peak_response = retrieval_cfg.get('merge_peak_response', True)  # merge peak response during nms
        param = retrieval_cfg.get('param', None)  # metric free parameters

        # process each peak
        instance_list = []
        for i in range(len(peak_response_maps)):
            class_idx = peak_list[i, 1]

            # extract hyper-params
            if isinstance(param, tuple):
                # shared param
                bg_threshold_factor, penalty_factor, balance_factor = param
            elif isinstance(param, list):
                # independent params between classes
                bg_threshold_factor, penalty_factor, balance_factor = param[class_idx]
            else:
                raise TypeError('Invalid hyper-params "%s".' % param)

            class_response = imresize(class_response_maps[class_idx], (img_height, img_width), interp='bicubic')
            bg_response = (class_response < bg_threshold_factor * class_response.mean()).astype(np.float32)
            peak_response_map = peak_response_maps[i]

            # select proposal
            max_val = -np.inf
            instance_mask = None

            for j in range(min(proposal_count, len(proposals))):
                raw_mask = imresize(proposals[j].astype(int), peak_response_map.shape, interp='nearest')
                # get contour of the proposal
                contour_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_GRADIENT, np.ones((contour_width, contour_width), np.uint8)).astype(bool)
                mask = raw_mask.astype(bool)
                # metric
                mask_area = mask.sum()
                if (mask_area >= proposal_size_limit[1] * img_area) or \
                        (mask_area < proposal_size_limit[0] * img_area):
                    continue
                else:
                    val = balance_factor * peak_response_map[mask].sum() + \
                        peak_response_map[contour_mask].sum() - \
                        penalty_factor * bg_response[mask].sum()
                    if val > max_val:
                        max_val = val
                        instance_mask = mask

            if instance_mask is not None:
                instance_list.append((max_val, class_idx, instance_mask, peak_response_map))

        instance_list = sorted(instance_list, key=lambda x: x[0], reverse=True)
        if nms_threshold is not None:
            instance_list = instance_nms(sorted(instance_list, key=lambda x: x[0], reverse=True), nms_threshold, merge_peak_response)
        return [dict(category=v[1], mask=v[2], prm=v[3]) for v in instance_list]
