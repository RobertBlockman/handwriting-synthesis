import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
import os
import logging
import numpy as np
import svgwrite
import drawing
from rnn import rnn
import textwrap
from tqdm import tqdm 
import subprocess
import random
import datetime

def trim_svg(svg_file):
    command = [
        "inkscape",
        "--batch-process",
        "--actions=select-all;transform-rotate:-90;export-area-drawing;export-margin:0",
        "-o", svg_file,
        svg_file
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print("An error occurred while trimming the SVG:", e)

def wrap_text(text, max_width):
    """
    Wraps text to ensure each line does not exceed max_width characters.
    Preserves empty lines.
    """
    wrapped_lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():  # Check if the line is empty
            wrapped_lines.append("")  # Preserve empty lines
        else:
            lines = textwrap.wrap(paragraph, width=max_width)
            wrapped_lines.extend(lines)
    return wrapped_lines

class Hand(object):

    def __init__(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir='logs',
            checkpoint_dir='checkpoints',
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.nn.restore()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        # Neural Network Inference for batch processing
        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )

        # Post-processing of Output for the batch
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]

        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        scale_factor = 1.5  # Scaling positions and dimensions up by 2x

        stroke_colors = stroke_colors or ['black'] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)  # Keeping stroke widths the same

        line_height = 25 * scale_factor
        top_offset = 40 * scale_factor
        view_width = 1500 * scale_factor
        view_height = line_height * (len(lines) + 1) + top_offset

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)

        current_height = top_offset
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
            if line.strip() == '':
                current_height += line_height
                continue

            # Calculate variances for each line
            height_variance = random.uniform(0, 4) * scale_factor
            margin_variance = random.uniform(-5, 5) * scale_factor
            rotation_variance = random.uniform(-0.5, 0.5)  # Rotation variance doesn't need scaling

            # Apply height variance
            current_height += height_variance

            # Prepare and draw each line with applied variances
            offsets[:, :2] *= 1.5 * scale_factor  # Scaling the offsets
            strokes = drawing.offsets_to_coords(offsets)
            strokes = drawing.denoise(strokes)
            strokes[:, :2] = drawing.align(strokes[:, :2])
            strokes[:, 1] *= -1

            # Construct the path with the margin variance
            path_string = "M{},{} ".format(margin_variance, 0)
            prev_eos = 1.0
            for x, y, eos in zip(*strokes.T):
                path_string += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x + margin_variance, y)
                prev_eos = eos

            # Create the path object and apply stroke (without scaling the width) and rotation
            path = svgwrite.path.Path(path_string)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            path.rotate(rotation_variance, center=(margin_variance, current_height))
            path.translate(0, current_height)

            # Add the path to the drawing
            dwg.add(path)

            # Increment the height for the next line
            current_height += line_height

        # Save the final drawing
        dwg.save()

    def write_multiple_svgs(self, content, max_height=4800):
        lines = wrap_text(content, 65)

        # Calculate the total height of the SVG
        line_height = 25 * 3.0  # Line height scaled by factor of 3.0
        top_offset = 40 * 3.0
        total_height = len(lines) * line_height + top_offset

        # Split content and generate SVGs if total height exceeds max_height
        if total_height > max_height:
            parts = self.split_content_at_paragraph(content, max_height, line_height, top_offset)
            for i, part in enumerate(parts):
                svg_filename = f'./img/output_part_{i+1}_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}-{datetime.datetime.now().second}.svg'
                self.write_single_svg(part, svg_filename)
        else:
            svg_filename = f'./img/output_{datetime.datetime.now().hour}-{datetime.datetime.now().minute}-{datetime.datetime.now().second}.svg'
            self.write_single_svg(content, svg_filename)

    def write_single_svg(self, content, filename):
        lines = wrap_text(content, 65)
        biases = [1 for _ in lines]
        stroke_widths = [1 for _ in lines]
        styles = [7 for _ in lines]  # Set the same style for all lines 2 for schizo, 7 for normal

        self.write(filename, lines, biases=biases, styles=styles, stroke_widths=stroke_widths)
        trim_svg(filename)

    def split_content_at_paragraph(self, content, max_height, line_height, top_offset):
        paragraphs = content.split('\n\n')
        parts = []
        current_part = ""
        current_height = top_offset

        for para in paragraphs:
            lines = wrap_text(para, 65)
            para_height = len(lines) * line_height

            if current_height + para_height > max_height and current_part:
                parts.append(current_part)
                current_part = para
                current_height = top_offset + para_height
            else:
                current_part += "\n\n" + para
                current_height += para_height

        if current_part.strip():
            parts.append(current_part)

        return parts

if __name__ == '__main__':
    hand = Hand()

    content = """
Blessed are the poor in spirit, for theirs is the kingdom of heaven.
Blessed are those who mourn, for they will be comforted.
Blessed are the meek, for they will inherit the earth.
Blessed are those who hunger and thirst for righteousness, for they will be filled.
Blessed are the merciful, for they will be shown mercy.
Blessed are the pure in heart, for they will see God.
Blessed are the peacemakers, for they will be called children of God.
Blessed are those who are persecuted because of righteousness, for theirs is the kingdom of heaven.
Blessed are you when people insult you, persecute you and falsely say all kinds of evil against you because of me. Rejoice and be glad, because great is your reward in heaven, for in the same way they persecuted the prophets who were before you.
Matthew 5:1-12

Therefore I tell you, do not worry about your life, what you will eat or drink; or about your body, what you will wear. Is not life more than food, and the body more than clothes? Look at the birds of the air; they do not sow or reap or store away in barns, and yet your heavenly Father feeds them. Are you not much more valuable than they? Can any one of you by worrying add a single hour to your life? And why do you worry about clothes? See how the flowers of the field grow. They do not labor or spin. Yet I tell you that not even Solomon in all his splendor was dressed like one of these. If that is how God clothes the grass of the field, which is here today and tomorrow is thrown into the fire, will he not much more clothe you—you of little faith? So do not worry, saying, ‘What shall we eat?’ or ‘What shall we drink?’ or ‘What shall we wear?’ For the pagans run after all these things, and your heavenly Father knows that you need them. But seek first his kingdom and his righteousness, and all these things will be given to you as well. Therefore do not worry about tomorrow, for tomorrow will worry about itself. Each day has enough trouble of its own.
Matthew 6:25-34

Ask and it will be given to you; seek and you will find; knock and the door will be opened to you. For everyone who asks receives; the one who seeks finds; and to the one who knocks, the door will be opened. Which of you, if your son asks for bread, will give him a stone? Or if he asks for a fish, will give him a snake? If you, then, though you are evil, know how to give good gifts to your children, how much more will your Father in heaven give good gifts to those who ask him! So in everything, do to others what you would have them do to you, for this sums up the Law and the Prophets.
Matthew 7:7-12
"""

    content = content.replace("’", "'")
    content = content.replace("“", '"')
    content = content.replace("”", '"')
    content = content.replace("`", '"')
    content = content.replace("`", '‘')
    content = content.replace("—", '-')
    content = content.replace("–", '-')
    content = content.replace("―", '-')
    #content = content.replace("Z", 'z')
    content = content.replace('\xa0', ' ')
    content = content.replace('*', ' ')
    content = content.replace('X', 'x')
    

    # Define the valid character set
    valid_chars = {'a', 'G', 'P', 'p', ',', '(', '#', '?', 'l', 'o', '7', 'e', ' ', 'L', 'K', '5', 'n', '1', 'q', 'I', 'f', 'j', 'O', 'u', 'M', 'w', '2', ')', 'h', 'A', ';', 'D', 'N', '.', 'Y', '9', 'T', 'm', 't', 'y', 'S', 's', 'k', '4', 'z', 'i', '!', 'r', 'd', 'H', 'C', '0', 'F', 'c', '3', 'B', '-', 'E', 'U', '8', 'V', 'W', '\x00', '6', ':', "'", 'R', '"', 'x', 'v', 'g', 'b', 'J'}

    # Iterate through each line and character
    for line_number, line in enumerate(content.split('\n'), 1):
        for char in line:
            if char not in valid_chars:
                print(f"Invalid character in line {line_number}: '{char}' (ASCII: {ord(char)})")


    for _ in range(0,5):
        hand.write_multiple_svgs(content)