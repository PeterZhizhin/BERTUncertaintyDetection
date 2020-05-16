import asyncio
import aiohttp_jinja2
import jinja2
from aiohttp import web
import argparse
import os

import model
import examples

routes = web.RouteTableDef()


@routes.view('/')
@aiohttp_jinja2.template('main.html')
class MainView(web.View):
    def _get_text_type(self, text, text_class):
        return {
            'text': text,
            'class': text_class,
        } if text else None

    def _convert_predictions_to_output(self, text, predictions):
        text_lines = text.split(os.linesep)

        # Find spans of lines
        lines_spans = [(0, len(text_lines[0]))]
        for text_line in text_lines[1:]:
            line_start = lines_spans[-1][1] + len(os.linesep)
            line_end = line_start + len(text_line)
            lines_spans.append((line_start, line_end))

        # For every line find all matching predictions O(|lines| * |predictions|) complexity,
        # may be implemented more efficiently.
        all_matching_predictions = [[] for _ in range(len(lines_spans))]
        for line_span, matching_predictions in zip(lines_spans, all_matching_predictions):
            for prediction in predictions:
                start = prediction['start']
                end = prediction['end']
                span_start, span_end = line_span
                if span_start <= start <= end <= span_end:
                    # Move the prediction span to the beginning of the line.
                    start -= span_start
                    end -= span_start
                    matching_predictions.append((start, end, prediction['type']))

        all_results = [[] for _ in range(len(lines_spans))]
        for text_line, predictions, result in zip(text_lines, all_matching_predictions, all_results):
            previous_prediction_end = 0
            for prediction in predictions:
                start, end, prediction_type = prediction

                head_text = self._get_text_type(text_line[previous_prediction_end:start], '')
                if head_text:
                    result.append(head_text)

                token_text = self._get_text_type(text_line[start:end], prediction_type)
                if token_text:
                    result.append(token_text)

                previous_prediction_end = end

            head_text = self._get_text_type(text_line[previous_prediction_end:len(text_line)], '')
            if head_text:
                result.append(head_text)

        return all_results

    async def get(self):
        request = self.request
        example_id = request.query.get('example_id')
        print('Got GET request with ID: {}'.format(request.query.get('example_id')))
        main_text = ''
        if example_id:
            example_id = int(example_id) - 1
            main_text = examples.examples[example_id]
        return {'main_text': main_text, 'examples_list': examples.titles}

    async def post(self):
        post_arguments = await self.request.post()
        main_text = post_arguments['main_text']
        print('Got POST request with text:\n{}'.format(main_text))

        model = self.request.config_dict['bert_model']
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, model.predict_text, main_text)

        formatted_text = self._convert_predictions_to_output(main_text, prediction)

        return {'main_text': main_text, 'output_text': formatted_text, 'examples_list': examples.titles}


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_path', required=True, type=str)
    argparser.add_argument('--labels_path', required=True, type=str)
    return argparser.parse_args()


def main():
    args = parse_args()

    app = web.Application()
    app['bert_model'] = model.Model(args.model_path, args.labels_path)

    app.add_routes(routes)
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates/'))
    web.run_app(app)


if __name__ == "__main__":
    main()
