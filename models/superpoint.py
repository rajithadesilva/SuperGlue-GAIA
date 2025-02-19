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

from pathlib import Path
import torch
from torch import nn
import numpy as np

import time
import csv

ENCDIM = 256

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim=256, bottleneck_dim=256-ENCDIM):
        super(LinearAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim))
        self.decoder = nn.Sequential(nn.Linear(bottleneck_dim, input_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class LinearEncoder(nn.Module):
    def __init__(self, input_dim=256, bottleneck_dim=256-ENCDIM):
        super(LinearEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, bottleneck_dim))

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
    
class EncoderDecoder(nn.Module):
    def __init__(self, encoded_dim=ENCDIM):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the Encoder-Only Network
class Encoder(nn.Module):
    def __init__(self, encoded_dim=ENCDIM):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

def find_nearest_masks_for_keypoints(masks, keypoints):
    N = masks.shape[0]
    result_indices = []
    points_with_255 = []

    for i in range(N):
            # Get coordinates of all points with value 255 in the current mask
            points_with_255.append(np.argwhere(masks[i] == 255))

    for keypoint in keypoints:
        x, y = keypoint
        min_distance = float('inf')
        nearest_mask_index = -1
        
        for i in range(N):            
            if points_with_255[i].size == 0:
                # If there are no points with 255 in this mask, skip it
                continue
            
            # Calculate the squared Euclidean distance to the keypoint for each point with value 255
            distances = np.sqrt((points_with_255[i][:, 0] - y) ** 2 + (points_with_255[i][:, 1] - x) ** 2)
            
            # Find the minimum distance in this mask
            min_dist_in_mask = np.min(distances)
            
            # Update the nearest mask and distance if the current one is closer
            if min_dist_in_mask < min_distance:
                min_distance = min_dist_in_mask
                nearest_mask_index = i
        
        # Store the nearest mask index for the current keypoint
        if min_distance < 2.0:
            result_indices.append(nearest_mask_index)
        else:
            result_indices.append(-1)
    
    return np.array(result_indices)

class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')
        #'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        '''
        self.LinearAutoencoder = LinearAutoencoder()
        self.LinearAutoencoder.load_state_dict(torch.load(f'./models/weights/{ENCDIM}_superpoint_encoder.pth'))
        self.LinearAutoencoder.to(self.device)
        self.LinearEncoder = LinearEncoder()
        autoencoder_state_dict = self.LinearAutoencoder.encoder.state_dict()
        encoder_state_dict = {}
        for key in autoencoder_state_dict:
            # Change key names from "0.weight" to "encoder.0.weight"
            new_key = 'encoder.' + key
            encoder_state_dict[new_key] = autoencoder_state_dict[key]
        # Load the modified state_dict into LinearEncoder
        self.LinearEncoder.load_state_dict(encoder_state_dict)
        self.LinearEncoder.to(self.device)

        print('Loaded SuperPoint Linear Autoencoder model')
        '''
        self.encdec = EncoderDecoder()
        self.encdec.load_state_dict(torch.load(f'./models/weights/{ENCDIM}_BN.pth'))
        self.encdec.to(self.device)

        self.semenc = Encoder().to(self.device)
        self.semenc.encoder.load_state_dict(self.encdec.encoder.state_dict())

        print('Loaded Semantic Encoder model')
        #'''
    def forward(self, data, masks):
        """ Compute keypoints, scores, descriptors for image """
        # TODO start keypoint extraction timer, this is just for one time not the image pair
        start_time_keypoint_extraction = time.time()

        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
               for k, d in zip(keypoints, descriptors)]        

        end_time_keypoint_extraction = time.time()

        elapsed_time_keypoint_extraction = end_time_keypoint_extraction - start_time_keypoint_extraction
        # print(f"keypoint extraction took {elapsed_time_keypoint_extraction:.6f} seconds")
        # Save to CSV
        with open("timer_keypoint_extraction.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time_keypoint_extraction])
        # TODO end keypoint extraction timer
        
        # TODO start KSI keypoint semantic integration timer
        start_time_ksi_keypoint_semantic_integration = time.time()

        #Modified to fit semantics from here
        if masks is not None:
            mask_indexes = find_nearest_masks_for_keypoints(masks, keypoints[0].cpu().numpy())

            embeddings = []

            for mask in masks:
                # TODO start semantic encoder timer
                start_time_semantic_encoder = time.time()

                mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
                mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(128, 128),  # Adjust to expected input size
                    mode='bilinear',
                    align_corners=False
                )
                emb = self.semenc(mask)
                embeddings.append(emb.cpu().numpy())

                end_time_semantic_encoder = time.time()

                elapsed_time_semantic_encoder = end_time_semantic_encoder - start_time_semantic_encoder
                # print(f"semantic_encoder took {elapsed_time_keypoint_extraction:.6f} seconds")
                # Save to CSV
                with open("timer_semantic_encoder.csv", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time_semantic_encoder])
                # TODO end semantic encoder timer

            semantic_descriptors = []
            for idx, desc in enumerate(descriptors[0].T):
                #'''
                if mask_indexes[idx]>= 0:
                    #desc = desc.to(self.device)
                    #reduced_desc = self.LinearEncoder(desc)
                    ##reduced_desc = torch.nn.functional.normalize(reduced_desc, p=2, dim=0)
                    #reduced_desc = reduced_desc.cpu().numpy()
                    #semantic_descriptors.append(np.concatenate((bottleneck_vector[0],reduced_desc)))
                    semantic_descriptors.append(desc.cpu().numpy()+embeddings[mask_indexes[idx]][0])
                    
                    
                else:
                #'''
                    semantic_descriptors.append(desc.cpu().numpy())
                    '''
                    desc = desc.to(self.device)
                    reduced_desc = self.LinearEncoder(desc)
                    #reduced_desc = torch.nn.functional.normalize(reduced_desc, p=2, dim=0)
                    reduced_desc = reduced_desc.cpu().numpy()

                    sem_background = np.any(masks, axis=0).astype(np.uint8)
                    sem_background = torch.tensor(sem_background, dtype=torch.float32).to(self.device)
                    sem_background = torch.nn.functional.interpolate(
                                        sem_background.unsqueeze(0).unsqueeze(0),
                                        size=(128, 128),  # Adjust to expected input size
                                        mode='bilinear',
                                        align_corners=False
                                    )
                    encoded = self.semenc(sem_background)
                    #encoded = torch.nn.functional.normalize(encoded, p=2, dim=0)
                    bottleneck_vector = (encoded.cpu().numpy())
                    semantic_descriptors.append(np.concatenate((bottleneck_vector[0],reduced_desc)))
                    '''
                
        else:
            mask_indexes = np.full((len(keypoints[0])), -1, dtype=np.int64)
            semantic_descriptors = descriptors[0].T.cpu()
        
        descriptors = np.array(semantic_descriptors,np.float32)
        descriptors = torch.tensor(descriptors, dtype=torch.float32).to(self.device).T.unsqueeze(0) #for unbranched
        #descriptors = torch.tensor(descriptors, dtype=torch.float32).to(self.device).unsqueeze(0) #for branched
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        mask_indexes = torch.tensor(mask_indexes, dtype=torch.int64).unsqueeze(0)

        end_time_ksi_keypoint_semantic_integration = time.time()

        elapsed_time_ksi_keypoint_semantic_integration = end_time_ksi_keypoint_semantic_integration - start_time_ksi_keypoint_semantic_integration
        # print(f"ksi_keypoint_semantic_integration took {elapsed_time_ksi_keypoint_semantic_integration:.6f} seconds")
        # Save to CSV
        with open("timer_ksi_keypoint_semantic_integration.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_time_ksi_keypoint_semantic_integration])
        # TODO END KSI keypoint semantic integration timer

        #Modified to fit semantics until here
        #mask_indexes = torch.tensor(mask_indexes, dtype=torch.int64).unsqueeze(0).to(self.device)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'indexes' : mask_indexes,
        }
