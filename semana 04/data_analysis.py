import os
import tqdm

import fiftyone as fo

import fiftyone.brain as fob
import fiftyone.zoo as foz


def create_samples(datasetpath):
    samples = list()

    classes = os.listdir(datasetpath)

    for class_name in classes:
        classdir = os.path.join(datasetpath, class_name)
        images = os.listdir(classdir)
        for imagename in tqdm.tqdm(images):
            sample = fo.Sample(filepath=os.path.join(classdir, imagename))
            sample['ground_truth'] = fo.Classification(label=class_name)
            samples.append(sample)
    
    return samples


if __name__ == "__main__":
    
    dataset = fo.Dataset("classification_dataset")
    
    DATAPATH = "./../data/train"
   
    samples = create_samples(DATAPATH)
                
    dataset.add_samples(samples)
    dataset.save()
    
    model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    embeddings = dataset.compute_embeddings(model)
    results = fob.compute_visualization(dataset, embeddings=embeddings, 
                                        seed=51, brain_key="img_viz")

    session = fo.launch_app(dataset, desktop=True)
    session.wait()