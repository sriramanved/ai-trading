"""
A module that provides an interface to send an email through Simple Mail Transfer Protocol (SMTP)
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config
import color_codes

def send_email(receiver_email_address, subject, body):
	"""
	Using SMTP, send an email to [receiver_email_address] with the given subject and body
	"""
	try:
		message = MIMEMultipart()
		message['From'] = config.SENDER_EMAIL_ADDRESS
		message['To'] = receiver_email_address
		message['Subject'] = subject
		message.attach(MIMEText(body, 'html'))
		with smtplib.SMTP('smtp.gmail.com', 587) as server:
			server.starttls()
			server.login(config.SENDER_EMAIL_ADDRESS, config.SENDER_APP_PASSWORD)
			server.sendmail(config.SENDER_EMAIL_ADDRESS, receiver_email_address, message.as_string())
			print(f'{color_codes.TEAL_COLOR_CODE}Update email sent successfully{color_codes.RESET_COLOR_CODE}')
	except Exception as e:
		print(f'{color_codes.RED_COLOR_CODE}Update email could not be sent{color_codes.RESET_COLOR_CODE}')