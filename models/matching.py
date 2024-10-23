# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import torch
import numpy as np
from .superpoint import SuperPoint
from .superglue import SuperGlue
#from .mdgat import MDGAT
from .utils import DimSqueeze, concatenate_dictionaries, reconstruct_predictions

class Matching(torch.nn.Module):
    """ Image Matching Frontend (Semantics + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))
        #self.mdgat = 1

    def forward(self, data, sem_background0=None, sem_background1=None):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}
        
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']}, sem_background0)
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
            
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']}, sem_background1)
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
        
        # Batch all features
        data = {**data, **pred}
        data['mask0'] = data['indexes0'] != -1  # Create mask0 based on indexes0 (background is -1)
        data['mask1'] = data['indexes1'] != -1  # Create mask1 based on indexes1 (background is -1)

        
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        #'''
        # Match all together
        pred = {**pred, **self.superglue(data)} 
        '''

        # Separate data based on indexes
        indexes0 = data['indexes0'].squeeze(0)
        indexes1 = data['indexes1'].squeeze(0)

        # Create masks for indexes0 and indexes1 separately
        mask_background0 = (indexes0 == -1)
        mask_background1 = (indexes1 == -1)
        mask_semantic0 = ~mask_background0
        mask_semantic1 = ~mask_background1

        data_background0 = {}
        data_background1 = {}
        data_semantic0 = {}
        data_semantic1 = {}
        for k, v in data.items():
            if torch.is_tensor(v) and '0' in k:
                if DimSqueeze(v) == indexes0.shape[0]:
                    data_background0[k] = v.squeeze(0)[mask_background0].unsqueeze(0)
                    data_semantic0[k] = v.squeeze(0)[mask_semantic0].unsqueeze(0)
                else:
                    data_background0[k] = v
                    data_semantic0[k] = v

            elif torch.is_tensor(v) and '1' in k:
                if DimSqueeze(v) == indexes1.shape[0]:
                    data_background1[k] = v.squeeze(0)[mask_background1].unsqueeze(0)
                    data_semantic1[k] = v.squeeze(0)[mask_semantic1].unsqueeze(0)
                else:
                    data_background1[k] = v
                    data_semantic1[k] = v

        # Ensure all tensors are on the same device
        device = indexes0.device
        for d in [data_background0, data_background1, data_semantic0, data_semantic1]:
            for k in d:
                d[k] = d[k].to(device)

        # Prepare data for background matches (indexes0 or indexes1 == -1)
        data_background = concatenate_dictionaries(data_background0, data_background1)
        if 'descriptors0' in data_background and 'descriptors1' in data_background:
            data_background['descriptors0'] = data_background['descriptors0'].squeeze(0).transpose(0, 1).unsqueeze(0)
            data_background['descriptors1'] = data_background['descriptors1'].squeeze(0).transpose(0, 1).unsqueeze(0)

        # Prepare data for semantic matches (indexes0 and indexes1 != -1)
        data_semantic = concatenate_dictionaries(data_semantic0, data_semantic1)
        if 'descriptors0' in data_semantic and 'descriptors1' in data_semantic:
            data_semantic['descriptors0'] = data_semantic['descriptors0'].squeeze(0).transpose(0, 1).unsqueeze(0)
            data_semantic['descriptors1'] = data_semantic['descriptors1'].squeeze(0).transpose(0, 1).unsqueeze(0)

        # Perform the matching for background data
        pred_background = self.superglue(data_background) if len(data_background) > 0 else {}
        # Perform the matching for semantic data
        pred_semantic = self.superglue(data_semantic) if len(data_semantic) > 0 else {}
        
        # Reconstruct the original prediction dictionary with matches
        pred = {**pred, ** reconstruct_predictions(pred, indexes0, indexes1, pred_background, pred_semantic)}
        '''
        return pred
    
'''
class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

	# Extract ORB (keypoints, scores, descriptors) if not provided
#add comment start here 
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}
#add comment end here 
        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred
'''
