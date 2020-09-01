import argparse
import json
from pathlib import Path
from string import Template

from tqdm import tqdm

template = Template(r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="description" content="$title">
    <meta name="author" content="Legg Yeung, Yonatan Bisk, Alex Polozov">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>$title</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" 
          integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" 
          integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery-bar-rating/1.2.2/themes/bars-movie.min.css" 
          integrity="sha512-8LZ4KH9KUaqOKvOgja1aeQWHUaz9hYy3vAOgijoHEpF3MD9Q2Zzq8A1DFNDK3JtPuoLm2JKSWSgCTHpQS+vWoQ==" 
          crossorigin="anonymous" />
    <style>
        footer { font-size: 80%; padding: 1rem; margin-top: 1rem; background: #dedede; }
    </style>
</head>
<body>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" 
            integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js" 
            integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js" 
            integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-bar-rating/1.2.2/jquery.barrating.min.js" 
            integrity="sha512-nUuQ/Dau+I/iyRH0p9sp2CpKY9zrtMQvDUG7iiVY8IBMj8ZL45MnONMbgfpFAdIDb7zS5qEJ7S056oE7f+mCXw==" 
            crossorigin="anonymous"></script>
            
    <div class="container">
        <h1 class="display-4">Welcome</h1>
        <p class="lead"> Thank you for your help in this study. </p>
        <p class="lead"> Please watch the video below. It shows an agent performing a common household task in a 
        virtual room. After the video, you will see the Instructions that describe the agent's actions. Please 
        evaluate these Instructions by answering two questions at the bottom of this page, and click "Send". </p>
        
        <h2>The agent's actions</h2>
        <video class="my-4" src="$video" width="256" height="256" controls></video>
        
        <h2>The agent's instructions</h2>
        <ol>$instructions</ol>
        
        <div class="my-5">
            <h4>Q1: The instructions accurately describe the actions in the video.</h4>
            Your assessment:
            <div class="py-4">
                <select id="q1-accurate" class="likert">
                    <option value="-2">Strongly disagree</option>
                    <option value="-1">Disagree</option>
                    <option value="0" selected>Neither agree nor disagree</option>
                    <option value="1">Agree</option>
                    <option value="2">Strongly agree</option>
                </select>
            </div>
        </div>
        
        <div class="my-5">
            <h4>Q2: The instructions are readable and understandable.</h4>
            Your assessment:
            <div class="py-4">
                <select id="q2-readable" class="likert">
                    <option value="-2">Strongly disagree</option>
                    <option value="-1">Disagree</option>
                    <option value="0" selected>Neither agree nor disagree</option>
                    <option value="1">Agree</option>
                    <option value="2">Strongly agree</option>
                </select>
            </div>
        </div>
        
        <div class="my-5">
            <button class="btn btn-outline-primary btn-lg" id="btnSend">
                <i class="fa fa-paper-plane" aria-hidden="true"></i> Send
            </button>
        </div>
    </div>
    
    <div id="consentModal" class="modal fade" tabindex="-1" data-backdrop="static" data-keyboard="false">
        <div class="modal-dialog modal-xl modal-dialog-scrollable modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Microsoft Research Project Participation Consent Form</h5>
                </div>
                <div class="modal-body">
                    <h5>INTRODUCTION</h5>
                    <p>Thank you for taking the time to consider volunteering in a Microsoft Corporation research 
                    project. This form explains what would happen if you join this research project. Please read it 
                    carefully and take as much time as you need. Email the study team to ask about anything that is not clear. Participation in this study is voluntary and you may withdraw at any time.</p>
                    <h5>TITLE OF RESEARCH PROJECT</h5>
                    <p>Alfred Speaks: Instruction Generation for Embodied Agents</p>
                    <h5>PURPOSE</h5>
                    <p>The goal of this study is to evaluate the accuracy and fluency of auto-generated instructions 
                    for embodied virtual agents.</p>
                    <h5>PROCEDURES</h5>
                    <p>During this project, the following will happen: you will first be provided instructions, 
                    a video of a virtual agent's actions, textual descriptions of these actions, and asked a series 
                    of questions about these descriptions. The study will take approximately 5-10 minutes to 
                    complete. We recommend running the study on the latest Microsoft Edge or Google Chrome browser. 
                    Microsoft will collect the numeric assessments you provide.</p>
                    <h5>PERSONAL INFORMATION</h5>
                    <p><strong>Personal information we collect:</strong> During the project we will not collect personal information about you. The data 
                    collection is anonymous, and we ask you to not disclose any personal information in the survey. We do collect an identifier that is provided as a parameter in the study URL but this identifier is not personally linked and is used only to authenticate your participation.</p>
                    <p><strong>How we use the data we collect:</strong> The data from the study will be used to conduct research and 
                    development into embodied virtual agents.</p>
                    <p><strong>How we store data:</strong> The data we collect will stored for a period of up to 18 months or less.</p>
                    <p>For additional information or concerns about how Microsoft handles personal information for 
                    Employees, External Staff and Candidates, please see the <a 
                    href="https://msdpn.azurewebsites.net/default?LID=62">Microsoft Global Data Protection Notice.</a></p>
                    <h5>BENEFITS AND RISKS</h5>
                    <p><strong>Benefits:</strong> There are no direct benefits to you that might reasonably be 
                    expected as a result of being in this study. The research team expects to better understand how 
                    to design systems that teach or explain the actions of an embodied virtual agent. You will receive 
                    any public benefit that may come these Research Results being shared with the greater scientific community.</p>
                    <p><strong>Risks:</strong> There are no anticipated, foreseeable risks or discomforts to you as a result of being in this study.</p>
                    <h5>PAYMENT FOR PARTICIPATION</h5>
                    <p>You will not be paid to take part in this study.</p>
                    <h5>CONTACT INFORMATION</h5>
                    <p>Should you have any questions concerning this project, or if you are injured as a result of 
                    being in this study, please contact Alex Polozov, at <a href="mailto:polozov@microsoft.com">polozov@microsoft.com</a>. Should you have any 
                    questions about your rights as a research subject, please contact Microsoft Research Ethics 
                    Program Feedback at <a href="mailto:MSRStudyfeedback@microsoft.com">MSRStudyfeedback@microsoft.com</a>.</p>
                    <h5>CONSENT</h5>
                    <p>By clicking "I agree" below, you confirm that the study was explained to you, you had a chance to ask questions before beginning the study, and all your questions were answered satisfactorily. By clicking "I agree" below, you voluntarily consent to participate, and you do not give up any legal rights you have as a study participant. You may request a link to download this form. On behalf of Microsoft, we thank you for your contribution and look forward to your research session.</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-primary" data-dismiss="modal">I agree</button>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="footer">
        <div class="container">
            <i class="fa fa-info-circle" aria-hidden="true"></i>
            <a target="_blank" href="http://go.microsoft.com/fwlink/?LinkId=518021">Microsoft Internal Data Privacy Notice</a>
            &nbsp; | &nbsp;
            <a target="_blank" href="https://www.microsoft.com/en-us/research/lab/microsoft-research-ai/">Microsoft Research AI</a>
            &nbsp; | &nbsp;
            &copy; Microsoft 2020
        </div>
    </footer>
    
    
    <script>
    $$(function() {
        $$('.likert').barrating({
            theme: 'bars-movie',
            deselectable: false,
        });
        
        $$('#consentModal').modal('show');
    });
    </script>
</body>
</html>
""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='instructions_20200817.json',
                        help='Path to the JSON file with instructions')
    parser.add_argument('-o', '--out', default='html',
                        help='Subdirectory for output HTML files')
    args = parser.parse_args()

    j = json.load(open(args.input))
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    for entry in tqdm(j):
        id = entry["id"]
        instructions = '\t'.join(f"<li>{s}</li>" for s in entry["generation"])
        html = template.substitute(title=id, video=entry["video_url"], instructions=instructions)
        with open(out_dir / f"{id}.html", 'w') as out:
            out.write(html)
