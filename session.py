import losswise
import time
import random


from contextlib import contextmanager


@contextmanager
def LWSession(api_key, tag, model_params, **session_params):
    # Set key
    losswise.set_api_key(api_key)

    # Create Session object
    session = losswise.Session(tag=tag, params=model_params, **session_params)

    try:
        yield session
    except Exception as ex:
        print("Problem with session encountered: {}".format(str(ex)))
    finally:
        session.done()


model_params = {'rnn_size': 512, "bla": "bla"}
with LWSession(api_key="EA8Q382M5", tag="simple_session_test", model_params=model_params) as session:
    graph = session.graph('loss', kind='min')
    for x in range(10):
        train_loss = 1. / (0.1 + x + 0.1 * random.random())
        test_loss = 1.5 / (0.1 + x + 0.2 * random.random())
        print(x)
        graph.append(x, {'train_loss': train_loss, 'test_loss': test_loss})
        time.sleep(5.)



