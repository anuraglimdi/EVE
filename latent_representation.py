"""
Script to load pre-trained model and get VAE latent space representation 
for sequences of interest, as inputs for other sequence -> function tasks.
"""

import os, sys
import json
import argparse
import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evol indices")
    parser.add_argument(
        "--MSA_data_folder", type=str, help="Folder where MSAs are stored"
    )
    parser.add_argument(
        "--MSA_list", type=str, help="List of proteins and corresponding MSA file name"
    )
    parser.add_argument(
        "--protein_index", type=int, help="Row index of protein in input mapping file"
    )
    parser.add_argument(
        "--MSA_weights_location",
        type=str,
        help="Location where weights for each sequence in the MSA will be stored",
    )
    parser.add_argument(
        "--theta_reweighting",
        type=float,
        help="Parameters for MSA sequence re-weighting",
    )
    parser.add_argument(
        "--VAE_checkpoint_location",
        type=str,
        help="Location where VAE model checkpoints will be stored",
    )
    parser.add_argument(
        "--model_name_suffix",
        default="Jan1",
        type=str,
        help="model checkpoint name is the protein name followed by this suffix",
    )
    parser.add_argument(
        "--model_parameters_location", type=str, help="Location of VAE model parameters"
    )
    parser.add_argument(
        "--latent_representation_location",
        type=str,
        help="Location where latent vector embedding of sequences in MSA will be saved",
    )
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.MSA_list)
    protein_name = mapping_file["protein_name"][args.protein_index]
    msa_location = (
        args.MSA_data_folder + os.sep + mapping_file["msa_location"][args.protein_index]
    )
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    if args.theta_reweighting is not None:
        theta = args.theta_reweighting
    else:
        try:
            theta = float(mapping_file["theta"][args.protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: " + str(theta))

    data = data_utils.MSA_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=args.MSA_weights_location
        + os.sep
        + protein_name
        + "_theta_"
        + str(theta)
        + ".npy",
    )

    model_name = protein_name + "_" + args.model_name_suffix
    print("Model name: " + str(model_name))

    model_params = json.load(open(args.model_parameters_location))

    model = VAE_model.VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=42,
    )
    model = model.to(model.device)

    try:
        checkpoint_name = (
            str(args.VAE_checkpoint_location) + os.sep + model_name + "_final"
        )
        checkpoint = torch.load(checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
    except:
        print("Unable to locate VAE model checkpoint")
        sys.exit(0)

    print("Getting latent representations for sequences")
    latent_reps = model.get_latent_representation(data=data, device=model.device)
