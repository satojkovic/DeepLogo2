import torch

if __name__ == '__main__':
    # Get pretrained weights
    checkpoint = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth',
        map_location='cpu',
        check_hash=True)

    # Remove class weights
    del checkpoint["model"]["class_embed.weight"]
    del checkpoint["model"]["class_embed.bias"]

    # Save
    torch.save(checkpoint,
               'detr-r50_no-class-head.pth')
