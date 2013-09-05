import sys, getopt
from os import listdir
from os.path import exists, isdir, isfile, join, basename
from subprocess import call
import re

def run_summarizer( docset_dir, output_dir ):
    """Run summarizer on all DUC 2002 document sets in docset_dir. Summaries are written to the output_dir."""

    summary_lengths = [ "50", "100", "200" ] # Make summaries of these lengths

    # summarize each dir in the docset dir
    for d in listdir( docset_dir ):
        dir_id = join( docset_dir, d )
        if( isdir( dir_id ) ):
            for l in summary_lengths:
                call( [ "python", "summarizer.py", "-d", dir_id, "-l", l, "-o", output_dir ] )


def make_eval_xml( settings_xml, systems_dir, models_dir ):
    """Create the settings.xml file for ROUGE evaluation Perl script. """

    # open and write xml header
    f = open( settings_xml, 'w' )
    f.write( '<ROUGE_EVAL version="1.55">\n' )

    # map of the different summarization experiments: exp[ docset id ][ summary size ] => a summary file
    exp = {}

    # organize the system(automatic) summarizer outputs
    # in order to compare different algorithms on the same docset/summary size
    for s in listdir( systems_dir ):
        fileid = join( systems_dir, s )
        if isfile( fileid ):
                          # algorithm|length|docset id
            m = re.search("([^.]+?)\.L(\d+)\.(d\d\d\d)", s )
            
            ds_id = m.groups()[ 2 ]
            size = int( m.groups()[ 1 ] )
            
            if ds_id not in exp:
                exp[ ds_id ] = {}
            if size not in exp[ ds_id ]:
                exp[ ds_id ][ size ] = []


            exp[ ds_id ][ size ].append( s )
             
    # for each experiment, write its xml showing where the system(automatic) summaries are 
    # and where the human model summaries are for comparison
    exp_num = 1            
    ds_ids = sorted( exp.keys(), reverse=True )
    for ds_id in ds_ids:
        print ds_id
        sizes = sorted( exp[ds_id].keys() )
        for size in sizes:
            print "\t{}".format(size)
            systems = sorted( exp[ ds_id ][ size ] )
            models = _find_models( models_dir, ds_id, size )        
            print "\t\t{}".format( exp[ds_id][size] )
            print
            print '\t\t{}'.format( models )
            
            f.write( '<EVAL ID="{}">\n'.format( exp_num ) ) 
            f.write( '<MODEL-ROOT>{}</MODEL-ROOT>\n'.format( models_dir ) )
            f.write( '<PEER-ROOT>{}</PEER-ROOT>\n'.format( systems_dir ) )
            f.write( '<INPUT-FORMAT TYPE="SEE"></INPUT-FORMAT>\n' )
            f.write( '<PEERS>\n' )            
            for i in range( len( systems ) ):
                system = systems[i]
                sysid = i+1 
                f.write( '<P ID="{}">{}</P>\n'.format( sysid, system ) ) 
            f.write( '</PEERS>\n' )

            f.write( '<MODELS>\n' ) 
            for i in range( len( models ) ):
                model = models[i]
                mid = i+1 
                f.write( '<M ID="{}">{}</M>\n'.format( mid, model ) ) 
            f.write( '</MODELS>\n' )
            f.write( '</EVAL>\n' )
    
            exp_num += 1
    
    f.write( '</ROUGE_EVAL>' )
    f.close()


def _find_models( models_dir, ds_id, size ):
    """Find human generated model summaries corresponding to system summaries (could be more than one)."""
    models = []

    # find all human summaries that match the docset id and length constraint in the models_dir
    for f in listdir( models_dir ):
                       # docset id |   length
        m = re.search( "(d\d\d\d)\.m\.(\d\d\d)", f )
        if m:
            m_ds_id = m.groups()[ 0 ]
            m_size = int( m.groups()[ 1 ] )
            
            if size == m_size and ds_id == m_ds_id:
                models.append( f )
    return models


def run_eval( settings_xml, rouge_dir ):
    """Evaluate summaries with ROUGE script."""
    print "ROUGE evaluation not implemented yet!"
    

def main( duc_dir, rouge_dir ):
    """Batch run summarizer on all DUC 2002 docsets for all evaluation lengths (50, 100, 200) and run ROUGE evaluation."""

    docsets_dir = join( duc_dir, 'docsets' )    # location of duc 2002 docset directory
    systems_dir = join( duc_dir, 'systems' )    # location of automatic system summarizer output directory
    models_dir = join( duc_dir, 'models' )      # location of human model summaries 
    settings_xml = join( duc_dir, 'settings.xml' )  # location to write settings.xml for ROUGE script

    # Run all summarizers 
    run_summarizer( docsets_dir, systems_dir )
    
    # make settings.xml for ROUGE 
    make_eval_xml( settings_xml, systems_dir, models_dir )
    
    # run ROUGE
    run_eval( settings_xml, rouge_dir )    


if __name__ == "__main__":

    script = basename( __file__ )
    argnames = '-d <ducdir> -r <rougedir>' 
    help_msg = "{} {}".format( script, argnames )

    duc_dir = ''
    rouge_dir = ''


    try:
        opts, args = getopt.getopt( sys.argv[1:], "hd:r:", ["duc-directory=", "rouge-directory="] )
    except getopt.GetoptError:
        print help_msg
        sys.exit(-1)
    for opt, arg in opts:
        if opt == '-h':
            print help_msg
            sys.exit(1)
        elif opt in ("-d", "--duc-directory"):
            duc_dir = arg
        elif opt in ("-r", "--rouge-directory"):
            rouge_dir = arg
                                                                                                                       
    if duc_dir == '' or not exists( duc_dir ):
        print help_msg
        print "Invalid DUC directory. Please check that DUC directory path."
        sys.exit(-1)
    if rouge_dir == '' or not exists( rouge_dir ):
        print help_msg
        print "Invalid ROUGE directory. Please check that ROUGE directory path."
        sys.exit(-1)

    main( duc_dir, rouge_dir )

