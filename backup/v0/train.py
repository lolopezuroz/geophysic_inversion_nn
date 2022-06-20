from tensor_dataset import load_dataset
from vgg_model import InversionModel
from parameters import args
import tensorflow as tf

# get training parameters
args = args
batchsize=args["batchsize"]
epochs=args["epochs"]
losses=args["losses"]
metrics=args["metrics"]
learning_rate=args["learning_rate"]
optimizer=args["optimizer"]
checkpoint_dir=args["checkpoint_dir"]
focus=args["focus"]

model = InversionModel()

# get dataset
train_dataset, test_dataset = load_dataset(args)
steps=len(train_dataset)


optimizer = optimizer(learning_rate=learning_rate)

loss_obj_cat, loss_obj_reg = losses["ice_occupation"],\
                             losses["ice_thickness"]

error_cat, error_reg = metrics["ice_occupation"],\
                       metrics["ice_thickness"]

"""model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    monitor='val_output_2_mse',
    mode='max',
    save_best_only=True)

model.compile(optimizer=optimizer,
              loss=[loss_obj_cat,loss_obj_reg],
              metrics=[error_cat, error_reg],
              loss_weights=[1.,0.])

model.fit(x=train_dataset,
          epochs=epochs,
          steps_per_epoch=steps,
          batch_size=batchsize,
          shuffle=True,
          validation_data=test_dataset,
          callbacks=[model_checkpoint_callback],
          verbose=2,
          )"""

@tf.function
def train_step(ice_velocity, ice_occupation, ice_thickness):
    with tf.GradientTape() as tape:
        logits = model(ice_velocity, training=True)
        loss_values = [loss_obj_cat(ice_occupation, logits[0]) * focus[0],loss_obj_reg(ice_thickness, logits[1]) * focus[0]]

    gradients = tape.gradient(loss_values, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for metric in metrics["ice_occupation"]: metric.update_state(ice_occupation, logits[0])
    for metric in metrics["ice_thickness"]: metric.update_state(ice_thickness, logits[1])

    return loss_values

@tf.function
def test_step(ice_velocity, ice_occupation, ice_thickness):
    logits = model(ice_velocity, training=False)
    for metric in metrics["ice_occupation"]: metric.update_state(ice_occupation, logits[0])
    for metric in metrics["ice_thickness"]: metric.update_state(ice_thickness, logits[1])

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (ice_velocity, ice_occupation, ice_thickness) in enumerate(train_dataset):
        loss_values = train_step(ice_velocity, ice_occupation, ice_thickness)
        if step % 20 == 0: print("\tTraining loss (for one batch) at step "+str(step)+" :",float(loss_values[0]), float(loss_values[1]))
    
    string = "(Training) "
    for output, product_metrics in metrics.items():
        string += output+" : "
        for metric in product_metrics:
            string += metric.name+" "+str(float(metric.result()))+" "
            metric.reset_states()
    print(string)
    
    for (ice_velocity, ice_occupation, ice_thickness) in test_dataset:
        test_step(ice_velocity, ice_occupation, ice_thickness)

    model.save_weights(checkpoint_dir)

    string = "(Validation) "
    for output, product_metrics in metrics.items():
        string += "| "+output+" : "
        for metric in product_metrics:
            string += metric.name+" "+str(float(metric.result()))+" "
            metric.reset_states()
    print(string)