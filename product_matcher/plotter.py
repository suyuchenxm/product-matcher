from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_problem1_hyperparameters(data, model):
	fig = make_subplots(
		rows=6,
		cols=1,
		horizontal_spacing=.5,
		shared_xaxes=True,
		subplot_titles=("Fitting Time", "Scoring Time", "Accuracy",
						"F1 score", "Precision", "Recall")
		)

	fig.add_trace(
		go.Bar(
			name='mean_fit_time',
			x=data['params_str'],
			y=data['mean_fit_time'],
			error_y=dict(type='data', array=data['std_fit_time'])
			),
		row=1, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_score_time',
			x=data['params_str'],
			y=data['mean_score_time'],
			error_y=dict(type='data', array=data['std_score_time'])
			),
		row=2, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_train_accuracy',
			x=data['params_str'],
			y=data['mean_train_accuracy'],
			error_y=dict(type='data', array=data['std_train_accuracy'])
			),
		row=3, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_test_accuracy',
			x=data['params_str'],
			y=data['mean_test_accuracy'],
			error_y=dict(type='data', array=data['std_test_accuracy'])
			),
		row=3, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_train_f1',
			x=data['params_str'],
			y=data['mean_train_f1'],
			error_y=dict(type='data', array=data['std_train_f1'])
			),
		row=4, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_test_f1',
			x=data['params_str'],
			y=data['mean_test_f1'],
			error_y=dict(type='data', array=data['std_test_f1'])
			),
		row=4, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_train_precision',
			x=data['params_str'],
			y=data['mean_train_precision'],
			error_y=dict(type='data', array=data['std_train_precision'])
			),
		row=5, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_test_precision',
			x=data['params_str'],
			y=data['mean_test_precision'],
			error_y=dict(type='data', array=data['std_test_precision'])
			),
		row=5, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_train_recall',
			x=data['params_str'],
			y=data['mean_train_recall'],
			error_y=dict(type='data', array=data['std_train_recall'])
			),
		row=6, col=1
		)
	fig.add_trace(
		go.Bar(
			name='mean_test_recall',
			x=data['params_str'],
			y=data['mean_test_recall'],
			error_y=dict(type='data', array=data['std_test_recall'])
			),
		row=6, col=1
		)
	fig.update_layout(
		barmode='group',
		height=1000,
		title_text=f"{model} hyper-parameters searching result"
		)

	return fig


def plot_problem1_trainingsize(data, model):
	fig = make_subplots(
		rows=6, cols=1, shared_xaxes=True,
		subplot_titles=("Fitting Time", "Scoring Time", "Accuracy",
						"F1 score", "Precision", "Recall")
		)

	fig.add_trace(
		go.Scatter(
			name='mean_fit_time',
			x=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
			   'i'],
			y=data["mean_fit_time"],
			),
		row=1, col=1
		)

	fig.add_trace(
		go.Scatter(
			x=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
			   'i'],
			name="mean_score_time",
			y=data["mean_score_time"],
			),
		row=2, col=1
		)

	fig.add_trace(
		go.Scatter(
			name="mean_train_accuracy",
			y=data["mean_train_accuracy"],
			),
		row=3, col=1
		)
	fig.add_trace(
		go.Scatter(
			name="mean_test_accuracy",
			y=data["mean_test_accuracy"],
			),
		row=3, col=1
		)

	fig.add_trace(
		go.Scatter(
			name="mean_train_f1",
			y=data["mean_train_f1"],
			),
		row=4, col=1
		)
	fig.add_trace(
		go.Scatter(
			name="mean_test_f1",
			y=data["mean_test_f1"],
			),
		row=4, col=1
		)

	fig.add_trace(
		go.Scatter(
			name="mean_train_precision",
			y=data["mean_train_precision"],
			),
		row=5, col=1
		)
	fig.add_trace(
		go.Scatter(
			name="mean_test_precision",
			y=data["mean_test_precision"],
			),
		row=5, col=1
		)

	fig.add_trace(
		go.Scatter(
			name="mean_train_recall",
			y=data["mean_train_recall"],
			),
		row=6, col=1
		)
	fig.add_trace(
		go.Scatter(
			name="mean_test_recall",
			y=data["mean_test_recall"],
			),
		row=6, col=1
		)

	fig.update_layout(
		xaxis={"type": 'category'},
		title_text=f"{model} Training Size Impact",
		height=1000
		)
	fig.update_xaxes(
		tickmode='array', tickvals=data['training_size'].index,
		ticktext=data['training_size'].values, )
	return fig
