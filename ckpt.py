torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discrminator_64': discrminator_64.state_dict(),
                'discrminator_128': discrminator_128.state_dict(),
                'discrminator_256': discrminator_256.state_dict(),
                'relation_classifier': relation_classifier.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'speech_encoder': speech_encoder.state_dict(),
                'optimizer_generator': optimizer_generator.state_dict(),
                'optimizer_discrminator': optimizer_discrminator.state_dict(),
                'optimizer_rs': optimizer_rs.state_dict(),
            })