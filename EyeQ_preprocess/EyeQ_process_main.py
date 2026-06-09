import time
import fundus_prep as prep
import glob
import os
import cv2 as cv
from PIL import ImageFile
from filter_missing_labels import filter_csv
from multiprocessing import Pool
from functools import partial


ImageFile.LOAD_TRUNCATED_IMAGES = True

def worker(image_path, save_path):
    dst_image = os.path.splitext(image_path.split('/')[-1])[0]+'.png'
    dst_path = os.path.join(save_path, dst_image)

    failure=""
    elapsed=0

    if os.path.exists(dst_path):
        print('already exists, continue...')
        ok = 0
        elapsed=0
        return ok, failure, elapsed
    try:
        t0 = time.time()
        
        img = prep.imread(image_path)
        r_img, borders, mask = prep.process_without_gb(img)
        r_img = cv.resize(r_img, (800, 800))
        prep.imwrite(dst_path, r_img)
        # mask = cv.resize(mask, (800, 800))
        # prep.imwrite(os.path.join('./original_mask', dst_image), mask)

        elapsed = time.time() - t0
        ok = 1
    except Exception as e:
        print(f'Error processing {image_path}: {type(e).__name__}: {e}')
        failure = (f"{image_path} | {e}")
        ok = 0

    return ok, failure, elapsed

def process(image_list, save_path):
    success = 0
    failures = []
    times = []

    with Pool(44) as p:
        worker_fn = partial(worker, save_path=save_path)
        for ok, failure, elapsed in p.imap_unordered(worker_fn, image_list, chunksize=32):
            if ok:
                success += 1
                times.append(elapsed)
            elif failure:
                failures.append(failure)

    return success, failures, times

"""
def process(image_list, save_path):
    success = 0
    failures = []
    times = []
    
    for image_path in image_list:
        dst_image = os.path.splitext(image_path.split('/')[-1])[0]+'.png'
        dst_path = os.path.join(save_path, dst_image)
        if os.path.exists(dst_path):
            print('continue...')
            continue
        try:
            t0 = time.time()
            
            img = prep.imread(image_path)
            r_img, borders, mask = prep.process_without_gb(img)
            r_img = cv.resize(r_img, (800, 800))
            prep.imwrite(dst_path, r_img)
            # mask = cv.resize(mask, (800, 800))
            # prep.imwrite(os.path.join('./original_mask', dst_image), mask)

            times.append(time.time() - t0)
            success += 1
        except Exception as e:
            print(f'Error processing {image_path}: {type(e).__name__}: {e}')
            failures.append(f"{image_path} | {e}")
            continue
    return success, failures, times
"""
def save_metrics(split, image_list, success, failures, times, split_time):
    os.makedirs('./metrics', exist_ok=True)
    avg_time = sum(times) / len(times) if times else 0

    with open(f'./metrics/{split}_metrics.txt', 'w') as f:
        f.write(f"Total de imagens encontradas : {len(image_list)}\n")
        f.write(f"Processadas com sucesso      : {success}\n")
        f.write(f"Falhas                       : {len(failures)}\n")
        f.write(f"Tempo total                  : {split_time:.1f}s\n")
        f.write(f"Tempo médio por imagem       : {avg_time:.3f}s\n")

    if failures:
        with open(f'./metrics/{split}_failures.txt', 'w') as f:
            f.write('\n'.join(failures))


if __name__ == "__main__":
    total_start = time.time()

    # treino
    train_image_list = glob.glob(os.path.join('./original_img/train', '*.jpeg'))
    train_save_path = prep.fold_dir('./train')

    train_start = time.time()
    success, failures, times = process(train_image_list, train_save_path)
    save_metrics('train', train_image_list, success, failures, times, time.time() - train_start)

    # teste
    test_image_list = glob.glob(os.path.join('./original_img/test', '*.jpeg'))
    test_save_path = prep.fold_dir('./test')

    test_start = time.time()
    success, failures, times = process(test_image_list, test_save_path)
    save_metrics('test', test_image_list, success, failures, times, time.time() - test_start)

    filter_csv('../data/Label_EyeQ_train.csv', './train', '../data/Label_EyeQ_train.filtered.csv')
    filter_csv('../data/Label_EyeQ_test.csv', './test', '../data/Label_EyeQ_test.filtered.csv')

    with open('./metrics/total_metrics.txt', 'w') as f:
        f.write(f"Tempo total geral: {time.time() - total_start:.1f}s\n")
