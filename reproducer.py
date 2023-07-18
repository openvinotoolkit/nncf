import pytest
import tensorflow as tf
import tensorflow_datasets as tfds


@pytest.mark.parametrize("devices", [1, 3, 2])
def test_distributed_fit(devices):
    datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = datasets['train'], datasets['test']

    if devices == 1:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    else:
        strategy = tf.distribute.MirroredStrategy([f"/gpu:{i}" for i in range(devices)])

    batch_size = 64 * strategy.num_replicas_in_sync
    train_dataset = mnist_test.cache().shuffle(10000).batch(batch_size)

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])

        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

    model.fit(train_dataset, epochs=1)


if __name__ == '__main__':
    test_distributed_fit(1)
    test_distributed_fit(3)
    test_distributed_fit(2)
